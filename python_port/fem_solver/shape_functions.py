"""Shape functions for 8-node quadrilateral elements"""
import numpy as np


def shape_function(xi: float, eta: float, element_mask: int) -> np.ndarray:
    """
    Compute shape functions at given point

    Node numbering: (3)--(6)--(2)
                    |         |
                    (7)       (5)
                    |         |
                    (0)--(4)--(1)
    """
    # Base shape functions (serendipity)
    N0 = (1 - xi) * (1 - eta) / 4
    N1 = (1 + xi) * (1 - eta) / 4
    N2 = (1 + xi) * (1 + eta) / 4
    N3 = (1 - xi) * (1 + eta) / 4
    N4 = (1 - xi**2) * (1 - eta) / 2  # Mid-side
    N5 = (1 - eta**2) * (1 + xi) / 2  # Mid-side
    N6 = (1 - xi**2) * (1 + eta) / 2  # Mid-side
    N7 = (1 - eta**2) * (1 - xi) / 2  # Mid-side

    base_N = np.array([N0, N1, N2, N3, N4, N5, N6, N7])

    # Filter by element mask
    nodes = []
    for i in range(8):
        if element_mask & (1 << i):
            nodes.append(base_N[i])
    return np.array(nodes)


def natural_derivatives(xi: float, eta: float, element_mask: int) -> np.ndarray:
    """Compute natural derivatives dN/dxi, dN/deta"""
    # Derivatives
    dN0_xi, dN0_eta = -(1-eta)/4, -(1-xi)/4
    dN1_xi, dN1_eta = (1-eta)/4, -(1+xi)/4
    dN2_xi, dN2_eta = (1+eta)/4, (1+xi)/4
    dN3_xi, dN3_eta = -(1+eta)/4, (1-xi)/4
    dN4_xi, dN4_eta = -xi*(1-eta), -(1-xi**2)/2
    dN5_xi, dN5_eta = (1-eta**2)/2, -eta*(1+xi)
    dN6_xi, dN6_eta = -xi*(1+eta), (1-xi**2)/2
    dN7_xi, dN7_eta = -(1-eta**2)/2, -eta*(1-xi)

    base_dN = np.array([
        [dN0_xi, dN1_xi, dN2_xi, dN3_xi, dN4_xi, dN5_xi, dN6_xi, dN7_xi],
        [dN0_eta, dN1_eta, dN2_eta, dN3_eta, dN4_eta, dN5_eta, dN6_eta, dN7_eta]
    ])

    nodes = []
    for i in range(8):
        if element_mask & (1 << i):
            nodes.append(i)
    return base_dN[:, nodes]


def jacobian(node_coords: np.ndarray, dN: np.ndarray) -> np.ndarray:
    """Compute Jacobian matrix J = dX/dxi"""
    return node_coords.T @ dN.T


def xy_derivatives(node_coords: np.ndarray, dN: np.ndarray) -> np.ndarray:
    """Compute derivatives in physical coordinates B = dN/dX"""
    J = jacobian(node_coords, dN)
    return np.linalg.inv(J) @ dN


def det_jacobian(node_coords: np.ndarray, dN: np.ndarray) -> float:
    """Compute determinant of Jacobian"""
    J = jacobian(node_coords, dN)
    return np.linalg.det(J)
