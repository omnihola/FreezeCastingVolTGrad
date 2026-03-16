# FreezeCasting Phase-Field Simulation (Python/FEniCS Port)

A Python port of the C++ phase-field simulation for modeling microstructural evolution in bioinspired materials from freeze-casting processes.

## Overview

This project implements a **phase-field finite element method (FEM)** with **adaptive mesh refinement (AMR)** for simulating solidification and microstructure evolution during freeze-casting. It models the coupled transport of solute (concentration) and the evolution of the solid-liquid interface.

## Features

- **Phase-Field Method**: Models solidification using the phase-field approach
- **Finite Element Method**: 8-node serendipity quadrilateral elements
- **Adaptive Mesh Refinement**: Custom quadtree-based AMR for efficient computation
- **Anisotropic Surface Energy**: Crystallographic anisotropy in the solidification front
- **Operator Splitting Scheme**: Biot-Savart time integration with BiCGSTAB solver

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/omnihola/FreezeCastingVolTGrad.git
cd FreezeCastingVolTGrad/python_port

# Create conda environment
conda env create -f conda_env.yml
conda activate freeze-casting-fem

# Or install dependencies manually
pip install -r requirements.txt

# Note: FEniCS must be installed via conda
conda install -c conda-forge dolfin
```

## Usage

### Running the Simulation

```bash
# With default parameters (for testing)
python main.py

# With custom input file
cp ../FreezeCasting_Flat/input ./
python main.py
```

### Input File Format

Create an `input` file with the following format:

```
11                  # maxLv - Maximum refinement level
1.0                 # gamma - Error tolerance for AMR
0.0002              # dt - Time step size
1000                # file_skip - Output frequency
100                 # mesh_skip - Mesh refinement frequency
10000000            # tmax - Maximum time steps
```

### Output Files

The simulation generates the following output files:

- `outPHI_<tstep>` - Phase field values
- `outU_<tstep>` - Concentration values
- `outX_<tstep>`, `outY_<tstep>` - Node coordinates
- `outPlot_<tstep>.dat` - Tecplot format for visualization

## Project Structure

```
python_port/
├── main.py                    # Entry point
├── requirements.txt          # Python dependencies
├── conda_env.yml             # Conda environment
├── fem_solver/
│   ├── __init__.py
│   ├── quadtree.py          # Custom AMR implementation
│   ├── shape_functions.py    # 8-node serendipity elements
│   ├── gauss_points.py      # Gaussian quadrature
│   ├── elements.py           # Element utilities
│   ├── fem.py               # FEM solver
│   ├── variables.py          # I/O operations
│   └── manager.py           # Main simulation loop
├── utils/
│   └── visualization.py      # Plotting utilities
└── tests/
    └── test_quadtree.py     # Unit tests
```

## Physics Model

### Governing Equations

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

### Numerical Method

- **Spatial Discretization**: FEM with 8-node quadrilateral elements
- **Temporal Discretization**: Operator splitting with W1L4/W1L6 weighting
- **Linear Solver**: BiCGSTAB
- **Mesh Adaptation**: Quadtree-based AMR with error-based refinement

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_quadtree.py::TestCoord -v
```

## Visualization

### Python Matplotlib

```python
from fem_solver.quadtree import QuadtreeMesh
from fem_solver.variables import read_input, pf_initialization
from utils.visualization import plot_field, plot_mesh

# After simulation
plot_field(node_coords, phi, "phi_field.png")
plot_mesh(node_coords, element_eft, "mesh.png")
```

### Tecplot

The output `.dat` files can be opened directly in Tecplot for advanced visualization.

### ParaView

VTK output can be generated for 3D visualization in ParaView.

## Development

### Adding New Features

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Implement changes with tests
3. Run tests: `pytest tests/ -v`
4. Commit and push

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public functions

## References

- Original C++ implementation: `FreezeCasting_Flat/`
- Phase-field method: [Your thesis.pdf]
- FEniCS: https://fenicsproject.org/

## License

MIT License

## Authors

- Ported from C++ by Claude Code

## Acknowledgments

Based on research in computational materials science for bioinspired materials from freeze-casting processes.
