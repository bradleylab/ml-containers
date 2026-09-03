"""Objective-driven ground-filter tuning.

The tuner never decides what a good ground surface is. It is handed an
objective -- constraints, one thing to maximize, things to report -- composed by
whoever is calling it, reads a vocabulary of measurements from `criteria`, runs
a parameter search against PDAL, and returns every evaluation so the caller can
re-reason and re-pick without re-running.
"""

__version__ = "0.1.0"
