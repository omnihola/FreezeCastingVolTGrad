"""Main simulation manager"""
import time
import numpy as np
from .quadtree import QuadtreeMesh, quadtree_initialization
from .variables import read_input, pf_initialization, output
from .fem import time_discretization, mesh_refinement


def run():
    """
    Main simulation loop

    This is the Python port of C++ Manager.cpp
    """
    print("Starting FreezeCasting Phase-Field Simulation...")

    # Read input parameters
    try:
        params = read_input("input")
    except FileNotFoundError:
        # Use default parameters for testing
        print("Warning: input file not found, using default parameters")
        params = {
            'maxLv': 3,
            'gamma': 1.0,
            'dt': 0.0002,
            'file_skip': 1000,
            'mesh_skip': 100,
            'tmax': 100,
            'Nx': 0.8 * 2**3,
            'Ny': 0.8 * 2**3
        }

    maxLv = params['maxLv']
    gamma = params['gamma']
    dt = params['dt']
    file_skip = params['file_skip']
    mesh_skip = params['mesh_skip']
    tmax = params['tmax']
    Nx = params['Nx']
    Ny = params['Ny']

    print(f"Parameters:")
    print(f"  maxLv = {maxLv}")
    print(f"  gamma = {gamma}")
    print(f"  dt = {dt}")
    print(f"  tmax = {tmax}")
    print(f"  Nx = {Nx}, Ny = {Ny}")

    # Initialize quadtree mesh (option 11 = initialization)
    print("Initializing mesh...")
    quadtree = quadtree_initialization(Nx, Ny, maxLv, gamma, 11)
    quadtree.add_nodes()
    node_coords, element_eft = quadtree.report()

    print(f"Mesh: {len(node_coords)} nodes, {len(element_eft)} elements")

    # Initialize phase field and concentration
    print("Initializing fields...")
    phi, u = pf_initialization(Nx, Ny, node_coords)
    phi_vel = np.zeros(len(node_coords))
    u_vel = np.zeros(len(node_coords))
    theta = np.zeros(len(node_coords))

    # Time stepping loop
    print("Starting time stepping...")
    start_time = time.time()

    for tloop in range(tmax):
        # Mesh refinement (periodic)
        if tloop % mesh_skip == 0 and tloop > 0:
            print(f"Mesh refinement at tstep {tloop}...")
            node_coords, phi, u, element_eft = mesh_refinement(
                phi, u, node_coords, element_eft, maxLv, gamma
            )
            phi_vel = np.zeros(len(node_coords))
            u_vel = np.zeros(len(node_coords))
            theta = np.zeros(len(node_coords))

        # Time discretization
        phi, u = time_discretization(
            phi, u, phi_vel, u_vel, dt,
            node_coords, element_eft
        )

        # Output
        if (tloop + 1) % file_skip == 0:
            output(tloop + 1, phi, u, theta, node_coords)
            elapsed = time.time() - start_time
            print(f"tstep = {tloop + 1}, elapsed = {elapsed:.1f}s")

        # Progress
        if (tloop + 1) % (file_skip // 10) == 0:
            elapsed = time.time() - start_time
            print(f"tstep = {tloop + 1}, time = {elapsed:.1f}s")

    print("All Done!")
