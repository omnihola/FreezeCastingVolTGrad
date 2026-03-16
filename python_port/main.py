#!/usr/bin/env python3
"""Entry point for FreezeCasting Phase-Field Simulation"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fem_solver.manager import run


if __name__ == "__main__":
    run()
