"""Input/Output operations for phase-field simulation"""
import numpy as np
from typing import List, Tuple, Dict
import os


def read_input(filename: str = "input") -> Dict:
    """
    Read simulation parameters from input file

    Format (from C++ Variables.cpp):
    maxLv
    gamma
    dt
    file_skip
    mesh_skip
    tmax
    """
    params = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

    params['maxLv'] = int(lines[0].split()[0])
    params['gamma'] = float(lines[1].split()[0])
    params['dt'] = float(lines[2].split()[0])
    params['file_skip'] = int(lines[3].split()[0])
    params['mesh_skip'] = int(lines[4].split()[0])
    params['tmax'] = int(lines[5].split()[0])

    # Derived parameters
    params['Nx'] = 0.8 * 2**params['maxLv']
    params['Ny'] = 0.8 * 2**params['maxLv']

    return params


def pf_initialization(nx: float, ny: float,
                     node_coords: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize phase field and concentration

    From C++ Variables.cpp lines 65-111:
    phi = -tanh((y - rad) / sqrt(2))
    rad = x_mid / 64
    x_mid = nx / 2
    """
    n_nodes = len(node_coords)
    phi = np.zeros(n_nodes)
    u = np.zeros(n_nodes)

    x_mid = nx / 2.0
    y_mid = ny / 2.0
    rad = x_mid / 64.0

    Cinf = 0.25
    k = 0.1
    Cl = 0.74

    for i, coord in enumerate(node_coords):
        # Initial condition: phi = -tanh((y - rad) / sqrt(2))
        dist = coord.y - rad
        phi[i] = -np.tanh(dist / np.sqrt(2))

        # Initial concentration (from C++ line 97)
        u[i] = 1 / (1 - k) * (2 * Cinf / (Cl * (1 + k - (1 - k) * -1)) - 1)

    return phi, u


def output(tstep: int, phi: np.ndarray, u: np.ndarray, theta: np.ndarray,
          node_coords: List, output_dir: str = ".") -> None:
    """
    Write output files in Tecplot and VTK format

    From C++ Variables.cpp lines 113-187
    """
    # Tecplot format files
    phi_file = os.path.join(output_dir, f"outPHI_{tstep}")
    u_file = os.path.join(output_dir, f"outU_{tstep}")
    x_file = os.path.join(output_dir, f"outX_{tstep}")
    y_file = os.path.join(output_dir, f"outY_{tstep}")
    plot_file = os.path.join(output_dir, f"outPlot_{tstep}.dat")

    # Write PHI
    with open(phi_file, 'w') as f:
        for val in phi:
            f.write(f"{val:.15e}\n")

    # Write U
    with open(u_file, 'w') as f:
        for val in u:
            f.write(f"{val:.15e}\n")

    # Write X coordinates
    with open(x_file, 'w') as f:
        for coord in node_coords:
            f.write(f"{coord.x:.15e}\n")

    # Write Y coordinates
    with open(y_file, 'w') as f:
        for coord in node_coords:
            f.write(f"{coord.y:.15e}\n")

    # Write Tecplot format
    k = 0.1
    Cl = 0.74

    with open(plot_file, 'w') as f:
        f.write('VARIABLES = "X", "Y", "PHI", "U", "C", "Theta"\n')
        f.write(f'ZONE T = "{tstep}"\n')

        for i, coord in enumerate(node_coords):
            if 0 <= coord.x <= 204.8:
                C = Cl / 2 * (u[i] * (1 - k) + 1) * (1 + k - (1 - k) * phi[i])
                f.write(f"{coord.x} {coord.y} {phi[i]} {u[i]} {C} {theta[i]}\n")

    print(f"Output written: tstep = {tstep}")
