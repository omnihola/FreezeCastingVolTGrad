"""Visualization utilities for phase-field results"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from typing import List


def plot_field(node_coords: List, phi: np.ndarray,
              filename: str = "field.png", title: str = "Phase Field"):
    """
    Plot phase field as contourf

    Args:
        node_coords: List of Coord objects
        phi: Phase field values
        filename: Output filename
        title: Plot title
    """
    x = np.array([c.x for c in node_coords])
    y = np.array([c.y for c in node_coords])

    plt.figure(figsize=(10, 8))
    plt.tricontourf(x, y, phi, levels=50, cmap='RdBu')
    plt.colorbar(label="PHI")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.aspect('equal')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_mesh(node_coords: List, element_eft: List[List[int]],
             filename: str = "mesh.png", title: str = "Finite Element Mesh"):
    """
    Plot finite element mesh

    Args:
        node_coords: List of Coord objects
        element_eft: Element freedom table (list of node index lists)
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    for elem in element_eft:
        # Close the element by appending first node
        xs = [node_coords[i].x for i in elem] + [node_coords[elem[0]].x]
        ys = [node_coords[i].y for i in elem] + [node_coords[elem[0]].y]
        plt.plot(xs, ys, 'k-', linewidth=0.5)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.aspect('equal')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_comparison(node_coords: List, phi: np.ndarray, u: np.ndarray,
                   filename: str = "comparison.png"):
    """
    Plot both PHI and U side by side
    """
    x = np.array([c.x for c in node_coords])
    y = np.array([c.y for c in node_coords])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # PHI
    im1 = ax1.tricontourf(x, y, phi, levels=50, cmap='RdBu')
    ax1.set_title("Phase Field (PHI)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    # U (concentration)
    im2 = ax2.tricontourf(x, y, u, levels=50, cmap='viridis')
    ax2.set_title("Concentration (U)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
