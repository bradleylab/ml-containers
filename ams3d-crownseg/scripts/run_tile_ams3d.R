#!/usr/bin/env Rscript
# Per-tile AMS3D worker for the Tyson production pipeline.
#
# Handles one LAZ tile end-to-end:
#   1. Read + height-normalize
#   2. Canopy filter (Z >= MIN_HEIGHT)
#   3. AMS3D via crownsegmentr::segment_tree_crowns()
#   4. Centroid-in-core filter (keep only trees whose centroid is in the
#      tile's 100 m core, not the 10 m buffer — prevents double-counting
#      across tile boundaries)
#   5. Global tree ID assignment: encodes (grid_x, grid_y, local_id) into
#      a single 64-bit integer so IDs never collide across tiles
#   6. Micro-cluster filtering (< MIN_REAL_PTS points → treeID = NA)
#   7. Emit LAZ + parquet attribute table + JSON log
#
# Reference R pattern: ../../lidr_baseline/scripts/run_watershed_crownseg.R

suppressPackageStartupMessages({
  library(lidR)
  library(crownsegmentr)
  library(data.table)
  library(optparse)
  library(arrow)
  library(jsonlite)
})

# --- Constants (verified tile geometry, see plan file) ---
# tile_X_Y UTM minimum corner: (X*100 + OFFSET_X, Y*100 + OFFSET_Y)
# Each tile is 120 m × 120 m. Inner 100 m × 100 m is the "core";
# outer 10 m on each side is the buffer overlapping with neighbors.
OFFSET_X <- 713449.02
OFFSET_Y <- 4265570.08
TILE_SIZE <- 120
BUFFER <- 10
MIN_HEIGHT <- 5           # m above ground for canopy filter
MIN_REAL_PTS <- 500       # min points per tree to count as "real"
GROUND_CATCHALL_ID <- 2147483647L  # crownsegmentr's ground/low-veg bucket
CL2TH <- 0.4              # crown_length_to_tree_height (fixed)

# --- CLI args ---
opts <- parse_args(OptionParser(option_list = list(
  make_option("--tile-name", type = "character"),
  make_option("--grid-x",    type = "integer"),
  make_option("--grid-y",    type = "integer"),
  make_option("--cd2th",     type = "numeric", default = 0.4),
  make_option("--convergence", type = "numeric", default = 0.3),
  make_option("--in-dir",    type = "character"),
  make_option("--out-dir",   type = "character")
)))

# `optparse` converts dashes to underscores in names
tile_name <- opts$`tile-name`
grid_x    <- as.integer(opts$`grid-x`)
grid_y    <- as.integer(opts$`grid-y`)
cd2th     <- opts$cd2th
convergence <- opts$convergence
in_dir    <- opts$`in-dir`
out_dir   <- opts$`out-dir`

stopifnot(!is.null(tile_name), !is.null(grid_x), !is.null(grid_y),
          !is.null(in_dir),    !is.null(out_dir))

t_start <- Sys.time()
log_data <- list(
  tile_name = tile_name, grid_x = grid_x, grid_y = grid_y,
  cd2th = cd2th, convergence = convergence,
  started_at = format(t_start, "%Y-%m-%dT%H:%M:%S%z")
)

# --- Tile core bounds ---
core_minx <- grid_x * 100 + OFFSET_X + BUFFER
core_maxx <- grid_x * 100 + OFFSET_X + BUFFER + 100
core_miny <- grid_y * 100 + OFFSET_Y + BUFFER
core_maxy <- grid_y * 100 + OFFSET_Y + BUFFER + 100

# --- Paths ---
laz_in    <- file.path(in_dir, paste0(tile_name, ".laz"))
laz_out   <- file.path(out_dir, "laz",     paste0(tile_name, ".laz"))
pqt_out   <- file.path(out_dir, "parquet", paste0(tile_name, "_trees.parquet"))
log_out   <- file.path(out_dir, "logs",    paste0(tile_name, ".json"))

dir.create(dirname(laz_out), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(pqt_out), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(log_out), recursive = TRUE, showWarnings = FALSE)

write_log <- function(extra = list()) {
  writeLines(toJSON(c(log_data, extra), auto_unbox = TRUE, pretty = TRUE), log_out)
}

# --- Read + normalize ---
cat("[1/7] reading", laz_in, "\n")
if (!file.exists(laz_in)) {
  write_log(list(status = "error", error = "input LAZ missing"))
  stop("input LAZ not found: ", laz_in)
}
las <- readLAS(laz_in)
log_data$n_input_points <- npoints(las)

if (is.empty(las)) {
  cat("empty LAS\n")
  write_log(list(status = "empty"))
  quit(status = 0)
}

# Guard against ultra-sparse tiles: TIN height normalization (step 2) requires
# at least 3 points to triangulate. Seen in practice on edge-of-flight tiles
# with 1-2 lidar returns (e.g. tile_-11_26, 1 pt, 2026-04-15 production run).
if (npoints(las) < 3L) {
  cat(sprintf("below-minimum points (%d < 3) — marking sparse\n", npoints(las)))
  write_log(list(status = "below_minimum", n_input_points = npoints(las)))
  quit(status = 0)
}

if (sum(las$Classification == 2L) == 0) {
  cat("[2a/7] no ground class; running CSF classifier\n")
  las <- classify_ground(las, algorithm = csf())
}

# TIN height normalization triangulates the ground surface from class-2 points
# only. <3 ground points breaks that even when total point count is OK. Seen
# on sparse edge tiles (e.g. tile_-7_-2 with 33 pts / 1 ground from PMF).
n_ground <- sum(las$Classification == 2L)
if (n_ground < 3L) {
  cat(sprintf("below-minimum ground points (%d < 3) — marking sparse\n", n_ground))
  write_log(list(status = "below_minimum_ground", n_input_points = npoints(las),
                 n_ground_points = n_ground))
  quit(status = 0)
}

cat("[2/7] normalizing heights (TIN)\n")
nlas <- normalize_height(las, algorithm = tin())

canopy_mask <- nlas$Z >= MIN_HEIGHT
log_data$n_canopy_points <- sum(canopy_mask)

if (log_data$n_canopy_points < MIN_REAL_PTS) {
  cat("insufficient canopy points — marking sparse\n")
  write_log(list(status = "sparse"))
  quit(status = 0)
}

canopy_df <- data.frame(
  x = nlas$X[canopy_mask],
  y = nlas$Y[canopy_mask],
  z = nlas$Z[canopy_mask]
)

# --- AMS3D ---
cat("[3/7] running AMS3D (CD2TH=", cd2th, ", convergence=", convergence, ")\n", sep = "")
t_seg <- Sys.time()
res <- tryCatch(
  segment_tree_crowns(
    canopy_df,
    crown_diameter_to_tree_height = cd2th,
    crown_length_to_tree_height = CL2TH,
    centroid_convergence_distance = convergence
  ),
  error = function(e) {
    cat("segment_tree_crowns failed:", e$message, "\n")
    write_log(list(status = "error", error = paste("segmentation:", e$message)))
    quit(status = 1)
  }
)
seg_seconds <- as.numeric(difftime(Sys.time(), t_seg, units = "secs"))
log_data$segmentation_seconds <- round(seg_seconds, 1)

# --- Centroid-in-core filter + global ID assignment ---
cat("[4/7] computing centroids and filtering to core\n")
# Keep dt aligned with canopy_df row order so the per-point assignment
# later in step 5 matches 1:1. Only filter out ground catch-all when
# computing centroids, not when building the per-point treeID vector.
dt <- as.data.table(res)

centroids <- dt[crown_id != GROUND_CATCHALL_ID, .(
  centroid_x = mean(x), centroid_y = mean(y), centroid_z = mean(z),
  height = max(z), n_points = .N
), by = crown_id]

# Core membership
centroids[, is_core := centroid_x >= core_minx & centroid_x < core_maxx &
                      centroid_y >= core_miny & centroid_y < core_maxy]
# Real tree flag (>= MIN_REAL_PTS and in core)
centroids[, is_real := is_core & n_points >= MIN_REAL_PTS]

log_data$n_total_clusters  <- nrow(centroids)
log_data$n_core_clusters   <- sum(centroids$is_core)
log_data$n_real_trees      <- sum(centroids$is_real)

# Global ID: (grid_x + 128) << 48 | (grid_y + 128) << 32 | local_id
# Using R's bit64 integer64 via the bitwShiftL at 32-bit only, so instead
# encode as string "gx_gy_localid" for portability. Numeric global_id
# computed by Python in post-processing if needed.
centroids[, tile_local_id := crown_id]
centroids[, global_id := sprintf("%d_%d_%d", grid_x, grid_y, tile_local_id)]

# --- Build per-point treeID for the LAZ output ---
# Only canopy points that are part of a real, core-owned tree get a treeID.
# dt is row-aligned with canopy_df (no filtering done above), so assignment
# to full_tree_ids at canopy_positions is 1:1.
cat("[5/7] building per-point treeID\n")
real_ids <- centroids[is_real == TRUE, crown_id]
dt[, treeID_local := fifelse(crown_id %in% real_ids, as.integer(crown_id), NA_integer_)]

full_tree_ids <- rep(NA_integer_, npoints(las))
canopy_positions <- which(canopy_mask)
stopifnot(nrow(dt) == length(canopy_positions))
full_tree_ids[canopy_positions] <- dt$treeID_local

# --- Emit LAZ ---
cat("[6/7] writing LAZ\n")
las_out <- add_lasattribute(las, full_tree_ids, name = "treeID",
                             desc = "AMS3D local tree ID")
writeLAS(las_out, laz_out)

# --- Emit parquet ---
cat("[7/7] writing parquet attribute table\n")
attr_dt <- centroids[, .(
  global_id, tile_name = tile_name, grid_x = grid_x, grid_y = grid_y,
  tile_local_id, centroid_x, centroid_y, centroid_z,
  height, n_points, is_core, is_real
)]
write_parquet(attr_dt, pqt_out)

# --- Log + done ---
log_data$status <- "ok"
log_data$laz_out <- laz_out
log_data$parquet_out <- pqt_out
log_data$total_seconds <- round(as.numeric(difftime(Sys.time(), t_start, units = "secs")), 1)
log_data$finished_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
write_log()

cat(sprintf("DONE %s: %d real trees, %d clusters total, %.1fs\n",
            tile_name, log_data$n_real_trees, log_data$n_total_clusters,
            log_data$total_seconds))
