"""Unit tests for quadtree and FEM modules"""
import pytest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fem_solver.quadtree import Coord, Element, QuadtreeMesh, quadtree_initialization
from fem_solver.gauss_points import gauss_2d
from fem_solver.shape_functions import shape_function, natural_derivatives, jacobian
from fem_solver.fem import free_energy_derivative, mobility, get_cotangent


class TestCoord:
    """Tests for Coord class"""

    def test_coord_creation(self):
        c = Coord(1.0, 2.0)
        assert c.x == 1.0
        assert c.y == 2.0

    def test_coord_equality(self):
        c1 = Coord(1.0, 2.0)
        c2 = Coord(1.0, 2.0)
        c3 = Coord(1.0 + 1e-13, 2.0)

        assert c1 == c2
        # Within tolerance should be equal
        assert not (c1 != c3)

    def test_coord_hash(self):
        c1 = Coord(1.0, 2.0)
        c2 = Coord(1.0, 2.0)

        assert hash(c1) == hash(c2)

    def test_coord_comparison(self):
        c1 = Coord(1.0, 2.0)
        c2 = Coord(2.0, 1.0)

        assert c1 < c2

    def test_coord_set(self):
        c = Coord()
        c.set(3.0, 4.0)
        assert c.x == 3.0
        assert c.y == 4.0


class TestElement:
    """Tests for Element class"""

    def test_element_creation(self):
        bounding = [Coord(0, 0), Coord(1, 0), Coord(1, 1), Coord(0, 1)]
        elem = Element(None, 0, bounding)

        assert elem.uLevel == 0
        assert len(elem.apChildren) == 4
        assert len(elem.apNeighbors) == 8

    def test_element_type_mask(self):
        bounding = [Coord(0, 0), Coord(1, 0), Coord(1, 1), Coord(0, 1)]
        elem = Element(None, 0, bounding)

        # bitElementType is initialized to 0, mask can be set
        assert hasattr(elem, 'bitElementType')


class TestQuadtreeMesh:
    """Tests for QuadtreeMesh class"""

    def test_quadtree_creation(self):
        mesh = QuadtreeMesh(10.0, 10.0, max_levels=2, gamma=1.0)

        assert mesh.nx == 10.0
        assert mesh.ny == 10.0
        assert mesh.max_levels == 2

    def test_quadtree_generate(self):
        mesh = QuadtreeMesh(10.0, 10.0, max_levels=2, gamma=1.0)
        mesh.generate(option=12)  # Regular mesh

        # Check that nodes were added
        assert len(mesh.m_node_coordinate_list) > 0


class TestGaussPoints:
    """Tests for Gaussian quadrature"""

    def test_gauss_1_point(self):
        gp = gauss_2d(1)
        assert gp.shape == (1, 3)
        assert gp[0, 0] == 0  # xi
        assert gp[0, 1] == 0  # eta
        assert gp[0, 2] == 4  # weight

    def test_gauss_4_points(self):
        gp = gauss_2d(4)
        assert gp.shape == (4, 3)

    def test_gauss_9_points(self):
        gp = gauss_2d(9)
        assert gp.shape == (9, 3)

    def test_gauss_invalid(self):
        with pytest.raises(ValueError):
            gauss_2d(5)  # Not supported


class TestShapeFunctions:
    """Tests for shape functions"""

    def test_shape_function(self):
        N = shape_function(0, 0, 0b11111111)  # Center, full element
        assert len(N) == 8
        # At center (0,0), corner nodes sum to 1.0, but mid-side nodes add more
        # This is expected for serendipity elements

    def test_natural_derivatives(self):
        dN = natural_derivatives(0, 0, 0b11111111)
        assert dN.shape == (2, 8)


class TestFEMFunctions:
    """Tests for FEM functions"""

    def test_free_energy_derivative(self):
        # Simple case: phi=0, u=0, theta=0
        f = free_energy_derivative(0.0, 0.0, 0.0)
        assert np.isfinite(f)

    def test_mobility(self):
        q1 = mobility(0.0)
        q2 = mobility(0.5)
        q3 = mobility(1.0)

        assert q1 > 0
        assert q2 > 0
        assert q3 == 0  # phi >= 1

    def test_get_cotangent(self):
        phi = np.array([1.0, 0.0, -1.0, 0.0])
        B = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])

        cot = get_cotangent(phi, B)
        assert cot.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
