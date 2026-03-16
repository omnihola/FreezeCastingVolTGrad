"""FEniCS-based FEM solver for phase-field simulation"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import bicgstab
from typing import Tuple, List, Optional

# Physics constants (from C++ FEM.cpp lines 25-48)
EPSILON_4 = 0.05       # Anisotropy strength
FOLD = 4               # Crystal symmetry fold
ALPHA = 0.0            # Orientation angle (degrees)
CINF = 0.25            # Interface concentration
K = 0.1                # Partition coefficient
CL = 0.74              # Liquid concentration
A1 = 5.0 * np.sqrt(2.0) / 8.0
A2 = 47.0 / 75.0
EPSILON = 100.0        # W/d0 ratio


def get_cotangent(phi: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cotangent vector for anisotropy: cot = B @ phi"""
    return B @ phi


def free_energy_derivative(phi: float, u: float, theta: float,
                           lam: float = 1.0) -> float:
    """
    Compute f(phi, u, theta, lambda) from C++ FEM.cpp line 323
    f = phi*(1-phi^2) - lambda*(1-phi^2)^2*(u+theta+0.03*phi*(1-phi^2)*noise)
    """
    phi_sq = phi * phi
    one_minus_phi_sq = 1 - phi_sq
    noise = np.random.random() - 0.5
    return (phi * one_minus_phi_sq -
            lam * one_minus_phi_sq**2 *
            (u + theta + 0.03 * phi * one_minus_phi_sq * noise))


def mobility(phi: float, k: float = K) -> float:
    """
    Compute mobility function q(phi, k) from C++ FEM.cpp line 330
    q = (1 - phi) / (1 + k - (1 - k) * phi) for phi < 1, else 0
    """
    if phi >= 1:
        return 0.0
    return (1 - phi) / (1 + k - (1 - k) * phi)


def compute_element_matrices(node_coords: np.ndarray, phi_vals: np.ndarray,
                              u_vals: np.ndarray, theta_vals: np.ndarray,
                              dt: float, n_gauss: int = 9) -> Tuple:
    """
    Compute element mass and stiffness matrices for one element

    Returns: M11, M12, M21, M22, K11, K12, K21, K22, F1
    """
    from .gauss_points import gauss_2d
    from .shape_functions import shape_function, natural_derivatives, xy_derivatives, det_jacobian

    gp = gauss_2d(n_gauss)
    n_nodes = len(node_coords)

    # Initialize matrices
    M11 = np.zeros((n_nodes, n_nodes))
    M12 = np.zeros((n_nodes, n_nodes))
    M21 = np.zeros((n_nodes, n_nodes))
    M22 = np.zeros((n_nodes, n_nodes))
    K11 = np.zeros((n_nodes, n_nodes))
    K12 = np.zeros((n_nodes, n_nodes))
    K21 = np.zeros((n_nodes, n_nodes))
    K22 = np.zeros((n_nodes, n_nodes))
    F1 = np.zeros(n_nodes)

    # Element parameters
    Dtelda = A1 * A2 * EPSILON

    for i in range(n_gauss):
        xi, eta, w = gp[i]

        # Determine element mask based on number of nodes
        # 4 nodes = Q4 (0b1111), 5 nodes = Q5, etc.
        element_mask = (1 << n_nodes) - 1  # e.g., 4 nodes -> 0b1111 = 15
        N = shape_function(xi, eta, element_mask)
        dN = natural_derivatives(xi, eta, element_mask)

        # Physical coordinates and derivatives
        B = xy_derivatives(node_coords, dN)
        J = det_jacobian(node_coords, dN)

        # Values at Gauss point
        phi_gp = N @ phi_vals
        u_gp = N @ u_vals
        theta_gp = N @ theta_vals

        # Cotangent and angle for anisotropy
        cot = get_cotangent(phi_vals, B)
        derx, dery = cot[0], cot[1]
        angle = np.arctan2(dery, derx)

        # Anisotropy coefficients (from C++ lines 127-128)
        as_ = 1 + EPSILON_4 * np.cos(FOLD * (angle - ALPHA * np.pi / 180))
        asp = -FOLD * EPSILON_4 * np.sin(FOLD * (angle - ALPHA * np.pi / 180))

        # Element matrices (from C++ lines 136-144)
        M11 += (1 - (1 - K) * theta_gp) * as_**2 * np.outer(N, N) * w * J
        M21 += -(1 + (1 - K) * u_gp) * 0.5 * np.outer(N, N) * w * J
        M22 += ((1 + K) * 0.5 - (1 - K) * 0.5 * phi_gp) * np.outer(N, N) * w * J

        K11 += as_**2 * (B.T @ B) * w * J
        K11 += as_ * asp * (B[1:].T @ B[0:1] - B[0:1].T @ B[1:]) * w * J

        if derx**2 + dery**2 > 1e-12:
            K21 += (1 + (1 - K) * u_gp) * np.sqrt(np.abs(np.sum(phi_vals))) / np.sqrt(8.0 * (derx**2 + dery**2)) * (B.T @ B) * w * J

        K22 += Dtelda * mobility(phi_gp, K) * (B.T @ B) * w * J

        F1 += N * free_energy_derivative(phi_gp, u_gp, theta_gp, 1.0) * w * J

    return M11, M12, M21, M22, K11, K12, K21, K22, F1


def assemble_global_matrices(node_coords: List, element_eft: List[List[int]],
                           phi: np.ndarray, u: np.ndarray, theta: np.ndarray,
                           dt: float) -> Tuple:
    """Assemble global sparse matrices"""
    n_nodes = len(node_coords)
    n_elements = len(element_eft)

    # Initialize COO format lists
    rows, cols, data = [], [], []

    M11_data, M12_data, M21_data, M22_data = [], [], [], []
    K11_data, K12_data, K21_data, K22_data = [], [], [], []
    F1_data = []

    for e_idx, eft in enumerate(element_eft):
        # Get element node coordinates
        elem_nodes = np.array([[node_coords[i].x, node_coords[i].y] for i in eft])
        phi_e = np.array([phi[i] for i in eft])
        u_e = np.array([u[i] for i in eft])
        theta_e = np.array([theta[i] for i in eft])

        # Compute element matrices
        M11, M12, M21, M22, K11, K12, K21, K22, F1 = compute_element_matrices(
            elem_nodes, phi_e, u_e, theta_e, dt
        )

        # Assemble into global
        for i in range(len(eft)):
            for j in range(len(eft)):
                if abs(M11[i,j]) > 1e-10:
                    M11_data.append((eft[i], eft[j], M11[i,j]))
                if abs(M21[i,j]) > 1e-10:
                    M21_data.append((eft[i], eft[j], M21[i,j]))
                if abs(M22[i,j]) > 1e-10:
                    M22_data.append((eft[i], eft[j], M22[i,j]))
                if abs(K11[i,j]) > 1e-10:
                    K11_data.append((eft[i], eft[j], K11[i,j]))
                if abs(K21[i,j]) > 1e-10:
                    K21_data.append((eft[i], eft[j], K21[i,j]))
                if abs(K22[i,j]) > 1e-10:
                    K22_data.append((eft[i], eft[j], K22[i,j]))
            if abs(F1[i]) > 1e-10:
                F1_data.append((eft[i], F1[i]))

    # Build sparse matrices
    def build_sparse(data_list, shape):
        if not data_list:
            return csr_matrix(shape)
        rows, cols, vals = zip(*data_list)
        return csr_matrix((vals, (rows, cols)), shape=shape)

    mM11 = build_sparse(M11_data, (n_nodes, n_nodes))
    mM21 = build_sparse(M21_data, (n_nodes, n_nodes))
    mM22 = build_sparse(M22_data, (n_nodes, n_nodes))
    mK11 = build_sparse(K11_data, (n_nodes, n_nodes))
    mK21 = build_sparse(K21_data, (n_nodes, n_nodes))
    mK22 = build_sparse(K22_data, (n_nodes, n_nodes))

    vF1 = np.zeros(n_nodes)
    for i, val in F1_data:
        vF1[i] += val

    return mM11, mM21, mM22, mK11, mK21, mK22, vF1


def time_discretization(phi: np.ndarray, u: np.ndarray,
                       phi_vel: np.ndarray, u_vel: np.ndarray,
                       dt: float, node_coords: List,
                       element_eft: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time stepping using operator splitting scheme from C++ FEM.cpp

    Returns: Updated phi and u arrays
    """
    n_nodes = len(node_coords)

    # Time-stepping weights (from C++ lines 225-231)
    rho = 0.0
    rhos = 0.0
    W1L4 = 1 / (1 + rho)
    W2L5 = 1 / ((1 + rho) * (1 + rhos))
    W1L6 = (3 + rho + rhos - rho * rhos) / (2 * (1 + rho) * (1 + rhos))
    lambda4 = 1.0
    lambda5 = 1 / (1 + rhos)

    # Theta placeholder
    theta = np.zeros(n_nodes)

    # Assemble matrices
    mM11, mM21, mM22, mK11, mK21, mK22, vF1 = assemble_global_matrices(
        node_coords, element_eft, phi, u, theta, dt
    )

    # Simplified time stepping (placeholder for full scheme)
    # Full implementation would use operator splitting with BiCGSTAB

    # Compute residual
    M_total = mM11 + mM21 + mM22
    K_total = mK11 + mK21 + mK22

    # Simple explicit update
    try:
        dphi, _ = bicgstab(M_total, vF1 - K_total @ phi, maxiter=500)
    except:
        dphi = np.zeros(n_nodes)

    # Update with stabilization
    phi = phi + W1L4 * dphi * dt

    # Similar for u
    try:
        du, _ = bicgstab(mM22, -mK22 @ u, maxiter=500)
    except:
        du = np.zeros(n_nodes)

    u = u + W1L4 * du * dt

    phi_vel = dphi
    u_vel = du

    return phi, u


def mesh_refinement(phi: np.ndarray, u: np.ndarray,
                   node_coords: List, element_eft: List[List[int]],
                   max_level: int, gamma: float) -> Tuple:
    """
    Perform mesh refinement and interpolate solutions

    This is a placeholder - full implementation uses quadtree
    """
    # Full implementation would:
    # 1. Run quadtree mesh refinement
    # 2. Interpolate phi, u to new mesh
    # 3. Return new node_coords, phi, u, element_eft
    return node_coords, phi, u, element_eft
