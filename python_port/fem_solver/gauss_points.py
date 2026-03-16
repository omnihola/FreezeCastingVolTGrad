"""Gaussian quadrature points for numerical integration"""
import numpy as np


def gauss_2d(n_points: int) -> np.ndarray:
    """
    Return 2D Gaussian quadrature points and weights

    Args:
        n_points: Number of points (1, 4, 9, or 16)

    Returns:
        Array of shape (n_points, 3) with [xi, eta, weight]
    """
    if n_points == 1:
        return np.array([[0, 0, 4.0]])
    elif n_points == 4:
        a = 1.0 / np.sqrt(3.0)
        return np.array([
            [-a, -a, 1.0],
            [a, -a, 1.0],
            [a, a, 1.0],
            [-a, a, 1.0]
        ])
    elif n_points == 9:
        # From C++ GaussPoints.cpp lines 21-30
        a = np.sqrt(5.0/6.0)
        b = np.sqrt(5.0/24.0)
        w1 = 25.0/81.0
        w2 = 40.0/81.0
        w3 = 64.0/81.0
        return np.array([
            [-a, -a, w1], [-a, 0, w2], [-a, a, w1],
            [0, -a, w2], [0, 0, w3], [0, a, w2],
            [a, -a, w1], [a, 0, w2], [a, a, w1]
        ])
    elif n_points == 16:
        # 4x4 Gauss points - from C++ lines 32-49
        a = 0.861136311594052
        b = 0.339981043584856
        w_a = 0.1210029933
        w_b = 0.2268518519
        # Simplified - implement proper 4x4 grid
        xi = np.array([-a, -b, b, a])
        eta = np.array([-a, -b, b, a])
        w = np.array([w_a, w_b, w_b, w_a])
        points = []
        for i in range(4):
            for j in range(4):
                points.append([xi[i], eta[j], w[i] * w[j]])
        return np.array(points)
    else:
        raise ValueError(f"Unsupported number of Gauss points: {n_points}")
