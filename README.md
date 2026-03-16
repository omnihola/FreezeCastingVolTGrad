# Computational Phase-Field Modeling for Microstructural Evolution in Bioinspired Material from Freeze-Casting Process

This repository contains a phase-field finite element method (FEM) simulation for modeling microstructural evolution in bioinspired materials from freeze-casting processes.

## Project Structure

```
FreezeCastingVolTGrad/
├── FreezeCasting_Flat/     # Original C++ implementation
├── python_port/            # Python port (FEniCS)
├── thesis.pdf              # Academic thesis
└── README.md              # This file
```

## Implementations

### 1. C++ Implementation (`FreezeCasting_Flat/`)

The original C++ implementation using:
- **Phase-field method** for solidification modeling
- **Finite Element Method (FEM)** with 8-node quadrilateral elements
- **Adaptive Mesh Refinement (AMR)** using quadtree data structure
- **Eigen** library for linear algebra
- **BiCGSTAB** solver for sparse linear systems

### 2. Python Port (`python_port/`)

A complete Python port using:
- **FEniCS/DOLFIN** for FEM computations
- **NumPy/SciPy** for array operations and sparse solvers
- **Custom Quadtree AMR** ported from C++
- Full compatibility with C++ input/output formats

## Quick Start

### Python Version (Recommended)

```bash
cd python_port

# Create conda environment
conda env create -f conda_env.yml
conda activate freeze-casting-fem

# Run simulation
python main.py
```

### C++ Version

See `FreezeCasting_Flat/` directory for build instructions.

## Features

- **Phase-Field Method**: Models solidification using the phase-field approach
- **Finite Element Method**: 8-node serendipity quadrilateral elements
- **Adaptive Mesh Refinement**: Custom quadtree-based AMR for efficient computation
- **Anisotropic Surface Energy**: Crystallographic anisotropy in the solidification front
- **Operator Splitting Scheme**: Biot-Savart time integration with BiCGSTAB solver

## Physics Model

The simulation solves coupled PDEs for the phase field (φ) and concentration (u):

1. **Phase Field**: ∂φ/∂t = -M_φ * δF/δφ
2. **Concentration**: ∂u/∂t = D∇²u + sources

Where F is the free energy functional with anisotropic surface energy.

### Key Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| ε₄ | Anisotropy strength | 0.05 |
| m | Folding number (symmetry) | 4 |
| k | Partition coefficient | 0.1 |
| C∞ | Interface concentration | 0.25 |
| Cl | Liquid concentration | 0.74 |

## Documentation

- [Python Port README](./python_port/README.md) - Detailed Python implementation docs
- [Thesis](./thesis.pdf) - Academic documentation

## License

MIT License

## Authors

- Original C++ implementation
- Python port by Claude Code
