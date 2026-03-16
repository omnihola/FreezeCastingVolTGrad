"""Element utilities for FEM mesh generation"""
import numpy as np


def node_coordinates(dx: float, nx: int, ny: int) -> np.ndarray:
    """Generate node coordinates for regular mesh"""
    n_nodes = (nx + 1) * (ny + 1)
    coords = np.zeros((n_nodes, 2))
    for i in range(nx + 1):
        for j in range(ny + 1):
            idx = i * (ny + 1) + j
            coords[idx, 0] = i * dx
            coords[idx, 1] = j * dx
    return coords


def element_nodes(nx: int, ny: int) -> np.ndarray:
    """Generate element connectivity"""
    n_elements = nx * ny
    elements = np.zeros((n_elements, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            elements[idx, 0] = i * (ny + 1) + j
            elements[idx, 1] = (i + 1) * (ny + 1) + j
            elements[idx, 2] = (i + 1) * (ny + 1) + j + 1
            elements[idx, 3] = i * (ny + 1) + j + 1
    return elements
