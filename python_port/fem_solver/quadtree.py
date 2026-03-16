"""Quadtree-based Adaptive Mesh Refinement for Phase-Field Simulation"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


# Node numbering convention:
# (3)--(6)--(2)
#  |         |
# (7)       (5)
#  |         |
# (0)--(4)--(1)


@dataclass
class Coord:
    """2D coordinate class for mesh nodes"""
    x: float = -1.0
    y: float = -1.0

    def __hash__(self):
        return hash((round(self.x, 12), round(self.y, 12)))

    def __eq__(self, other):
        if not isinstance(other, Coord):
            return False
        return abs(self.x - other.x) < 1e-12 and abs(self.y - other.y) < 1e-12

    def __lt__(self, other):
        if abs(self.x - other.x) > 1e-12:
            return self.x < other.x
        elif abs(self.y - other.y) > 1e-12:
            return self.y < other.y
        return False

    def set(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


@dataclass
class Element:
    """Quad tree element for adaptive mesh refinement

    Attributes:
        uLevel: refinement level
        acNodalCoordinates: 8 node coordinates [SW, SE, NE, NW, midS, midE, midN, midW]
        bitElementType: 8-bit mask for node types
        pMe: reference to self
        pParent: parent element
        apChildren: 4 child elements
        apNeighbors: 8 neighboring elements
        tolerance: error tolerance
    """
    pParent: Optional['Element'] = None
    uLevel: int = 0
    acNodalCoordinates: List[Coord] = field(default_factory=lambda: [Coord() for _ in range(8)])
    bitElementType: int = 0  # bitset<8>
    pMe: Optional['Element'] = None
    apChildren: List[Optional['Element']] = field(default_factory=lambda: [None] * 4)
    apNeighbors: List[Optional['Element']] = field(default_factory=lambda: [None] * 8)
    tolerance: float = 0.0

    def __post_init__(self):
        # Set self-reference
        self.pMe = self

    def subdivide(self, max_lv: int, vvp_level_element_list: List[List['Element']],
                  m_phi_coordinate_list: Dict[Coord, float],
                  m_u_coordinate_list: Dict[Coord, float],
                  m_phi_velocity_coordinate_list: Dict[Coord, float],
                  m_u_velocity_coordinate_list: Dict[Coord, float]) -> None:
        """Subdivide element into 4 children if conditions are met"""

        if self.apChildren[0] is None and self.check_neighbors(
            max_lv, vvp_level_element_list, m_phi_coordinate_list,
            m_u_coordinate_list, m_phi_velocity_coordinate_list,
            m_u_velocity_coordinate_list):

            child_neighbors = [None] * 8
            u_child_level = self.uLevel + 1

            # Create sub-elements
            # Child 0 (SW quadrant)
            sw = Coord(self.acNodalCoordinates[0].x, self.acNodalCoordinates[0].y)
            se = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      self.acNodalCoordinates[0].y)
            ne = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            nw = Coord(self.acNodalCoordinates[0].x,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            bounding = [sw, se, ne, nw]
            self.apChildren[0] = create_element(
                bounding, self, u_child_level, max_lv, vvp_level_element_list)

            # Interpolate values for new nodes
            if se not in m_phi_coordinate_list:
                m_phi_coordinate_list[se] = 0.5 * (m_phi_coordinate_list[self.acNodalCoordinates[0]] +
                                                    m_phi_coordinate_list[self.acNodalCoordinates[1]])
                m_u_coordinate_list[se] = 0.5 * (m_u_coordinate_list[self.acNodalCoordinates[0]] +
                                                  m_u_coordinate_list[self.acNodalCoordinates[1]])
                m_phi_velocity_coordinate_list[se] = 0.5 * (
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[0]] +
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[1]])
                m_u_velocity_coordinate_list[se] = 0.5 * (
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[0]] +
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[1]])

            if ne not in m_phi_coordinate_list:
                m_phi_coordinate_list[ne] = 0.25 * (
                    m_phi_coordinate_list[self.acNodalCoordinates[0]] +
                    m_phi_coordinate_list[self.acNodalCoordinates[1]] +
                    m_phi_coordinate_list[self.acNodalCoordinates[2]] +
                    m_phi_coordinate_list[self.acNodalCoordinates[3]])
                m_u_coordinate_list[ne] = 0.25 * (
                    m_u_coordinate_list[self.acNodalCoordinates[0]] +
                    m_u_coordinate_list[self.acNodalCoordinates[1]] +
                    m_u_coordinate_list[self.acNodalCoordinates[2]] +
                    m_u_coordinate_list[self.acNodalCoordinates[3]])
                m_phi_velocity_coordinate_list[ne] = 0.25 * (
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[0]] +
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[1]] +
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[2]] +
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[3]])
                m_u_velocity_coordinate_list[ne] = 0.25 * (
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[0]] +
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[1]] +
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[2]] +
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[3]])

            if nw not in m_phi_coordinate_list:
                m_phi_coordinate_list[nw] = 0.5 * (m_phi_coordinate_list[self.acNodalCoordinates[0]] +
                                                    m_phi_coordinate_list[self.acNodalCoordinates[3]])
                m_u_coordinate_list[nw] = 0.5 * (m_u_coordinate_list[self.acNodalCoordinates[0]] +
                                                  m_u_coordinate_list[self.acNodalCoordinates[3]])
                m_phi_velocity_coordinate_list[nw] = 0.5 * (
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[0]] +
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[3]])
                m_u_velocity_coordinate_list[nw] = 0.5 * (
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[0]] +
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[3]])

            # Child 1 (SE quadrant)
            sw = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      self.acNodalCoordinates[0].y)
            se = Coord(self.acNodalCoordinates[1].x, self.acNodalCoordinates[0].y)
            ne = Coord(self.acNodalCoordinates[1].x,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            nw = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            bounding = [sw, se, ne, nw]
            self.apChildren[1] = create_element(
                bounding, self, u_child_level, max_lv, vvp_level_element_list)

            if ne not in m_phi_coordinate_list:
                m_phi_coordinate_list[ne] = 0.5 * (m_phi_coordinate_list[self.acNodalCoordinates[1]] +
                                                    m_phi_coordinate_list[self.acNodalCoordinates[2]])
                m_u_coordinate_list[ne] = 0.5 * (m_u_coordinate_list[self.acNodalCoordinates[1]] +
                                                  m_u_coordinate_list[self.acNodalCoordinates[2]])
                m_phi_velocity_coordinate_list[ne] = 0.5 * (
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[1]] +
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[2]])
                m_u_velocity_coordinate_list[ne] = 0.5 * (
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[1]] +
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[2]])

            # Child 2 (NE quadrant)
            sw = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            se = Coord(self.acNodalCoordinates[1].x,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            ne = Coord(self.acNodalCoordinates[1].x, self.acNodalCoordinates[2].y)
            nw = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      self.acNodalCoordinates[2].y)
            bounding = [sw, se, ne, nw]
            self.apChildren[2] = create_element(
                bounding, self, u_child_level, max_lv, vvp_level_element_list)

            if nw not in m_phi_coordinate_list:
                m_phi_coordinate_list[nw] = 0.5 * (m_phi_coordinate_list[self.acNodalCoordinates[2]] +
                                                    m_phi_coordinate_list[self.acNodalCoordinates[3]])
                m_u_coordinate_list[nw] = 0.5 * (m_u_coordinate_list[self.acNodalCoordinates[2]] +
                                                  m_u_coordinate_list[self.acNodalCoordinates[3]])
                m_phi_velocity_coordinate_list[nw] = 0.5 * (
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[2]] +
                    m_phi_velocity_coordinate_list[self.acNodalCoordinates[3]])
                m_u_velocity_coordinate_list[nw] = 0.5 * (
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[2]] +
                    m_u_velocity_coordinate_list[self.acNodalCoordinates[3]])

            # Child 3 (NW quadrant)
            sw = Coord(self.acNodalCoordinates[0].x,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            se = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
            ne = Coord((self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                      self.acNodalCoordinates[2].y)
            nw = Coord(self.acNodalCoordinates[0].x, self.acNodalCoordinates[2].y)
            bounding = [sw, se, ne, nw]
            self.apChildren[3] = create_element(
                bounding, self, u_child_level, max_lv, vvp_level_element_list)

            # Set child neighbors
            # Child 0 neighbors
            child_neighbors[0] = self.apNeighbors[0]
            child_neighbors[1] = self.apNeighbors[0]
            child_neighbors[2] = self.apChildren[1]
            child_neighbors[3] = self.apChildren[1]
            child_neighbors[4] = self.apChildren[3]
            child_neighbors[5] = self.apChildren[3]
            child_neighbors[6] = self.apNeighbors[7]
            child_neighbors[7] = self.apNeighbors[7]
            self.apChildren[0].apNeighbors = child_neighbors.copy()

            # Child 1 neighbors
            child_neighbors[0] = self.apNeighbors[1]
            child_neighbors[1] = self.apNeighbors[1]
            child_neighbors[2] = self.apNeighbors[2]
            child_neighbors[3] = self.apNeighbors[2]
            child_neighbors[4] = self.apChildren[2]
            child_neighbors[5] = self.apChildren[2]
            child_neighbors[6] = self.apChildren[0]
            child_neighbors[7] = self.apChildren[0]
            self.apChildren[1].apNeighbors = child_neighbors.copy()

            # Child 2 neighbors
            child_neighbors[0] = self.apChildren[1]
            child_neighbors[1] = self.apChildren[1]
            child_neighbors[2] = self.apNeighbors[3]
            child_neighbors[3] = self.apNeighbors[3]
            child_neighbors[4] = self.apNeighbors[4]
            child_neighbors[5] = self.apNeighbors[4]
            child_neighbors[6] = self.apChildren[3]
            child_neighbors[7] = self.apChildren[3]
            self.apChildren[2].apNeighbors = child_neighbors.copy()

            # Child 3 neighbors
            child_neighbors[0] = self.apChildren[0]
            child_neighbors[1] = self.apChildren[0]
            child_neighbors[2] = self.apChildren[2]
            child_neighbors[3] = self.apChildren[2]
            child_neighbors[4] = self.apNeighbors[5]
            child_neighbors[5] = self.apNeighbors[5]
            child_neighbors[6] = self.apNeighbors[6]
            child_neighbors[7] = self.apNeighbors[6]
            self.apChildren[3].apNeighbors = child_neighbors.copy()

            # Update neighbors of parent neighbors
            # North neighbors (0 and 1)
            if self.apNeighbors[0] is not None and self.apNeighbors[1] is not None:
                if self.apNeighbors[0] != self.apNeighbors[1]:
                    self.apNeighbors[0].apNeighbors[4] = self.apChildren[0]
                    self.apNeighbors[0].apNeighbors[5] = self.apChildren[0]
                    self.apNeighbors[1].apNeighbors[4] = self.apChildren[1]
                    self.apNeighbors[1].apNeighbors[5] = self.apChildren[1]
                elif self.apNeighbors[0] is not None:
                    self.apNeighbors[0].apNeighbors[4] = self.apChildren[1]
                    self.apNeighbors[0].apNeighbors[5] = self.apChildren[0]

            # East neighbors (2 and 3)
            if self.apNeighbors[2] is not None and self.apNeighbors[3] is not None:
                if self.apNeighbors[2] != self.apNeighbors[3]:
                    self.apNeighbors[2].apNeighbors[6] = self.apChildren[1]
                    self.apNeighbors[2].apNeighbors[7] = self.apChildren[1]
                    self.apNeighbors[3].apNeighbors[6] = self.apChildren[2]
                    self.apNeighbors[3].apNeighbors[7] = self.apChildren[2]
                elif self.apNeighbors[2] is not None:
                    self.apNeighbors[2].apNeighbors[6] = self.apChildren[2]
                    self.apNeighbors[2].apNeighbors[7] = self.apChildren[1]

            # South neighbors (4 and 5)
            if self.apNeighbors[4] is not None and self.apNeighbors[5] is not None:
                if self.apNeighbors[4] != self.apNeighbors[5]:
                    self.apNeighbors[4].apNeighbors[0] = self.apChildren[2]
                    self.apNeighbors[4].apNeighbors[1] = self.apChildren[2]
                    self.apNeighbors[5].apNeighbors[0] = self.apChildren[3]
                    self.apNeighbors[5].apNeighbors[1] = self.apChildren[3]
                elif self.apNeighbors[4] is not None:
                    self.apNeighbors[4].apNeighbors[0] = self.apChildren[3]
                    self.apNeighbors[4].apNeighbors[1] = self.apChildren[2]

            # West neighbors (6 and 7)
            if self.apNeighbors[6] is not None and self.apNeighbors[7] is not None:
                if self.apNeighbors[6] != self.apNeighbors[7]:
                    self.apNeighbors[6].apNeighbors[2] = self.apChildren[3]
                    self.apNeighbors[6].apNeighbors[3] = self.apChildren[3]
                    self.apNeighbors[7].apNeighbors[2] = self.apChildren[0]
                    self.apNeighbors[7].apNeighbors[3] = self.apChildren[0]
                elif self.apNeighbors[6] is not None:
                    self.apNeighbors[6].apNeighbors[2] = self.apChildren[0]
                    self.apNeighbors[6].apNeighbors[3] = self.apChildren[3]

    def fuse(self, vvp_level_element_list: List[List['Element']], option: int,
             m_phi_coordinate_list: Dict[Coord, float],
             m_u_coordinate_list: Dict[Coord, float],
             m_phi_velocity_coordinate_list: Dict[Coord, float],
             m_u_velocity_coordinate_list: Dict[Coord, float],
             gamma: float) -> None:
        """Fuse children back into parent if error is small enough"""

        if (self.apChildren[0] is not None and
            self.apChildren[0].apChildren[0] is None and
            self.apChildren[1].apChildren[1] is None and
            self.apChildren[2].apChildren[2] is None and
            self.apChildren[3].apChildren[3] is None and
            not self.apChildren[0].check_error(option, m_phi_coordinate_list, m_u_coordinate_list, gamma) and
            not self.apChildren[1].check_error(option, m_phi_coordinate_list, m_u_coordinate_list, gamma) and
            not self.apChildren[2].check_error(option, m_phi_coordinate_list, m_u_coordinate_list, gamma) and
            not self.apChildren[3].check_error(option, m_phi_coordinate_list, m_u_coordinate_list, gamma) and
            not self.check_error(option, m_phi_coordinate_list, m_u_coordinate_list, gamma) and
            self.apChildren[0].apNeighbors[6] == self.apChildren[0].apNeighbors[7] and
            self.apChildren[0].apNeighbors[0] == self.apChildren[0].apNeighbors[1] and
            self.apChildren[1].apNeighbors[0] == self.apChildren[1].apNeighbors[1] and
            self.apChildren[1].apNeighbors[2] == self.apChildren[1].apNeighbors[3] and
            self.apChildren[2].apNeighbors[2] == self.apChildren[2].apNeighbors[3] and
            self.apChildren[2].apNeighbors[4] == self.apChildren[2].apNeighbors[5] and
            self.apChildren[3].apNeighbors[4] == self.apChildren[3].apNeighbors[5] and
            self.apChildren[3].apNeighbors[6] == self.apChildren[3].apNeighbors[7]):

            # Reset neighbors
            self.apNeighbors[0] = self.apChildren[0].apNeighbors[0]
            self.apNeighbors[1] = self.apChildren[1].apNeighbors[0]
            self.apNeighbors[2] = self.apChildren[1].apNeighbors[2]
            self.apNeighbors[3] = self.apChildren[2].apNeighbors[2]
            self.apNeighbors[4] = self.apChildren[2].apNeighbors[4]
            self.apNeighbors[5] = self.apChildren[3].apNeighbors[4]
            self.apNeighbors[6] = self.apChildren[3].apNeighbors[6]
            self.apNeighbors[7] = self.apChildren[0].apNeighbors[6]

            # Clear children
            for i in range(4):
                self.apChildren[i].clean()
                self.apChildren[i] = None

            # Update neighbor references
            # North neighbors (0 and 1)
            if (self.apNeighbors[0] is not None and self.apNeighbors[1] is not None and
                self.apNeighbors[0] != self.apNeighbors[1]):
                self.apNeighbors[0].apNeighbors[4] = self
                self.apNeighbors[0].apNeighbors[5] = self
                self.apNeighbors[1].apNeighbors[4] = self
                self.apNeighbors[1].apNeighbors[5] = self
            elif self.apNeighbors[0] is not None:
                self.apNeighbors[0].apNeighbors[4] = self
                self.apNeighbors[0].apNeighbors[5] = self

            # East neighbors (2 and 3)
            if (self.apNeighbors[2] is not None and self.apNeighbors[3] is not None and
                self.apNeighbors[2] != self.apNeighbors[3]):
                self.apNeighbors[2].apNeighbors[6] = self
                self.apNeighbors[2].apNeighbors[7] = self
                self.apNeighbors[3].apNeighbors[6] = self
                self.apNeighbors[3].apNeighbors[7] = self
            elif self.apNeighbors[2] is not None:
                self.apNeighbors[2].apNeighbors[6] = self
                self.apNeighbors[2].apNeighbors[7] = self

            # South neighbors (4 and 5)
            if (self.apNeighbors[4] is not None and self.apNeighbors[5] is not None and
                self.apNeighbors[4] != self.apNeighbors[5]):
                self.apNeighbors[4].apNeighbors[0] = self
                self.apNeighbors[4].apNeighbors[1] = self
                self.apNeighbors[5].apNeighbors[0] = self
                self.apNeighbors[5].apNeighbors[1] = self
            elif self.apNeighbors[4] is not None:
                self.apNeighbors[4].apNeighbors[0] = self
                self.apNeighbors[4].apNeighbors[1] = self

            # West neighbors (6 and 7)
            if (self.apNeighbors[6] is not None and self.apNeighbors[7] is not None and
                self.apNeighbors[6] != self.apNeighbors[7]):
                self.apNeighbors[6].apNeighbors[2] = self
                self.apNeighbors[6].apNeighbors[3] = self
                self.apNeighbors[7].apNeighbors[2] = self
                self.apNeighbors[7].apNeighbors[3] = self
            elif self.apNeighbors[6] is not None:
                self.apNeighbors[6].apNeighbors[2] = self
                self.apNeighbors[6].apNeighbors[3] = self

    def add_nodes(self, m_node_coordinate_list: Dict[Coord, int]) -> None:
        """Add nodes to coordinate list based on element type"""

        # Initialize bitElementType - set all bits to 1 (True)
        self.bitElementType = 0xFF  # All 8 bits set

        # Clear bits for edge nodes that share neighbors
        for i in range(4):
            if self.apNeighbors[2 * i] is self.apNeighbors[2 * i + 1]:
                # Clear bit at position i+4 (edge nodes)
                self.bitElementType &= ~(1 << (i + 4))

        # Assign nodal coordinates for edge nodes
        for i in range(8):
            if (self.bitElementType >> i) & 1:
                if i == 4:  # Mid-south
                    self.acNodalCoordinates[4].set(
                        (self.acNodalCoordinates[0].x + self.acNodalCoordinates[1].x) / 2,
                        self.acNodalCoordinates[0].y)
                elif i == 5:  # Mid-east
                    self.acNodalCoordinates[5].set(
                        self.acNodalCoordinates[1].x,
                        (self.acNodalCoordinates[1].y + self.acNodalCoordinates[2].y) / 2)
                elif i == 6:  # Mid-north
                    self.acNodalCoordinates[6].set(
                        (self.acNodalCoordinates[2].x + self.acNodalCoordinates[3].x) / 2,
                        self.acNodalCoordinates[2].y)
                elif i == 7:  # Mid-west
                    self.acNodalCoordinates[7].set(
                        self.acNodalCoordinates[0].x,
                        (self.acNodalCoordinates[0].y + self.acNodalCoordinates[3].y) / 2)

                coord = self.acNodalCoordinates[i]
                if coord not in m_node_coordinate_list:
                    m_node_coordinate_list[coord] = len(m_node_coordinate_list)

    def check_neighbors(self, max_lv: int, vvp_level_element_list: List[List['Element']],
                        m_phi_coordinate_list: Dict[Coord, float],
                        m_u_coordinate_list: Dict[Coord, float],
                        m_phi_velocity_coordinate_list: Dict[Coord, float],
                        m_u_velocity_coordinate_list: Dict[Coord, float]) -> bool:
        """Check if neighbors need subdivision"""

        for i in range(len(self.apNeighbors)):
            neighbor = self.apNeighbors[i]
            if neighbor is not None:
                if self.uLevel > neighbor.uLevel:
                    neighbor.subdivide(
                        max_lv, vvp_level_element_list,
                        m_phi_coordinate_list, m_u_coordinate_list,
                        m_phi_velocity_coordinate_list, m_u_velocity_coordinate_list)
        return True

    def check_error(self, option: int, m_phi_coordinate_list: Dict[Coord, float],
                    m_u_coordinate_list: Dict[Coord, float], gamma: float) -> bool:
        """Check if element needs refinement based on error criteria"""

        # Get midpoint for some options
        x_mid = max(m_phi_coordinate_list.keys(), key=lambda c: c.x).x / 2.0 if m_phi_coordinate_list else 102.4
        y_mid = max(m_phi_coordinate_list.keys(), key=lambda c: c.y).y / 2.0 if m_phi_coordinate_list else 102.4

        # Simplified error check for Python port
        # Case 10: Mesh refinement based on gradient
        if option == 10:
            # Calculate approximate gradient using nodal values
            dx = (self.acNodalCoordinates[1].x - self.acNodalCoordinates[0].x)
            dy = (self.acNodalCoordinates[2].y - self.acNodalCoordinates[0].y)

            # Estimate gradient from phi values if available
            if self.acNodalCoordinates[0] in m_phi_coordinate_list:
                phi_vals = [m_phi_coordinate_list.get(self.acNodalCoordinates[i], 0.0) for i in range(4)]
                # Simple gradient estimate
                der_x = (phi_vals[1] - phi_vals[0]) / dx if dx > 0 else 0
                der_y = (phi_vals[2] - phi_vals[0]) / dy if dy > 0 else 0
                grad_mag = np.sqrt(der_x**2 + der_y**2)

                return (grad_mag > 0.3 and
                        self.acNodalCoordinates[0].x >= 0 and
                        self.acNodalCoordinates[1].x <= 204.8 or
                        self.uLevel <= 3)

        # Case 11: Initialization
        if option == 11:
            rad = x_mid / 164.0
            # Use interface position check
            for i in range(4):
                y = self.acNodalCoordinates[i].y
                dist = y - rad
                psi_val = -np.tanh(dist / np.sqrt(2))

                # Check if near interface
                if abs(psi_val) < 0.99:
                    return True
            return self.uLevel <= 4

        # Case 12: Regular mesh - always refine
        if option == 12:
            return True

        # Default cases: function-based refinement
        f = [0.0] * 4
        for i in range(4):
            x = self.acNodalCoordinates[i].x
            y = self.acNodalCoordinates[i].y

            if option == 1:
                # Complex function
                f[i] = ((x - 5)**4 + (y - 7.5)**4 +
                        100 / ((x - 5)**4 + (2*(y - 7.5) - 2)**4 + (2*(y - 7.5) - 1)**2) +
                        100 / ((x - 5)**4 + (2*(y - 7.5) + 2)**4 + (2*(y - 7.5) + 1)**2) -
                        1 / (((y - 7.5) + 3)**4 + ((x - 5) / 15)**4) -
                        1 / (((y - 7.5) + 4)**4 + ((x - 5) / 15)**4) -
                        1 / (((y - 7.5) + 5)**4 + ((x - 5) / 15)**4) -
                        1 / (4 * ((x - 5) + (y - 7.5) + 4)**4 + (((x - 5) - (y - 7.5) + 1) / 5)**4))
            elif option == 2:
                f[i] = np.sin(x) - y + 2
            elif option == 3:
                f[i] = 1.0
            elif option == 4:
                dx = x - 2
                dy = y - 2
                f[i] = (dx**2 + dy**2 - 1)**3 - dx**2 * dy**3
            else:
                f[i] = 0.0

        # Check sign changes
        if (f[0] * f[1] <= 0 or f[1] * f[2] <= 0 or
            f[2] * f[3] <= 0 or f[3] * f[0] <= 0 or
            self.uLevel <= 0):
            return True

        return False

    def clean(self) -> None:
        """Clean up element references"""
        self.pMe = None
        self.pParent = None
        for i in range(4):
            self.apChildren[i] = None
        for i in range(8):
            self.apNeighbors[i] = None


class QuadtreeMesh:
    """Quadtree-based adaptive mesh for finite element method"""

    def __init__(self, nx: float = 204.8, ny: float = 204.8,
                 max_levels: int = 6, gamma: float = 0.1):
        """Initialize quadtree mesh

        Args:
            nx: Domain width
            ny: Domain height
            max_levels: Maximum refinement levels
            gamma: Error tolerance parameter
        """
        self.nx = nx
        self.ny = ny
        self.max_levels = max_levels
        self.gamma = gamma

        self.vvp_level_element_list: List[List[Element]] = []
        self.m_node_coordinate_list: Dict[Coord, int] = {}
        self.vv_eft: List[List[int]] = []
        self.vc_node_coordinates: List[Coord] = []

        # Coordinate maps for field values
        self.m_phi_coordinate_list: Dict[Coord, float] = {}
        self.m_u_coordinate_list: Dict[Coord, float] = {}
        self.m_phi_velocity_coordinate_list: Dict[Coord, float] = {}
        self.m_u_velocity_coordinate_list: Dict[Coord, float] = {}

    def generate(self, option: int = 12) -> None:
        """Generate quadtree mesh

        Args:
            option: Refinement criterion option
        """
        # Initialize
        quadtree_initialization(
            self.nx, self.ny, self.max_levels, self.gamma,
            self.vvp_level_element_list, self.m_node_coordinate_list,
            self.vv_eft, option,
            self.m_phi_coordinate_list, self.m_u_coordinate_list,
            self.m_phi_velocity_coordinate_list, self.m_u_velocity_coordinate_list)

        # Generate mesh (fuse and subdivide)
        self.mesh_generate(option)

        # Add nodes
        quadtree_add_nodes(self.vvp_level_element_list, self.m_node_coordinate_list)

        # Report elements
        self.report()

    def mesh_generate(self, option: int = 12) -> None:
        """Generate mesh through fuse and subdivide iterations"""
        quadtree_mesh_generate(
            self.max_levels, self.gamma,
            self.vvp_level_element_list, option,
            self.m_phi_coordinate_list, self.m_u_coordinate_list,
            self.m_phi_velocity_coordinate_list, self.m_u_velocity_coordinate_list)

    def report(self) -> None:
        """Generate final element and node report"""
        vp_final_element_list = []
        report_element(
            self.vvp_level_element_list, vp_final_element_list,
            self.m_node_coordinate_list, self.vv_eft,
            self.vc_node_coordinates)

    def get_node_coordinates(self) -> np.ndarray:
        """Get node coordinates as numpy array"""
        coords = np.zeros((len(self.vc_node_coordinates), 2))
        for i, coord in enumerate(self.vc_node_coordinates):
            coords[i, 0] = coord.x
            coords[i, 1] = coord.y
        return coords

    def get_node_coords_list(self) -> List[Coord]:
        """Get node coordinates as list of Coord objects"""
        return self.vc_node_coordinates

    def get_elements(self) -> List[Element]:
        """Get leaf elements"""
        elements = []
        for level_list in self.vvp_level_element_list:
            for elem in level_list:
                if elem.apChildren[0] is None:  # Leaf element
                    elements.append(elem)
        return elements

    def get_element_count(self) -> int:
        """Get total number of leaf elements"""
        return sum(1 for _ in self.get_elements())

    def get_node_count(self) -> int:
        """Get total number of nodes"""
        return len(self.m_node_coordinate_list)


def create_element(bounding: List[Coord], parent: Optional[Element],
                   u_level: int, max_lv: int,
                   vvp_level_element_list: List[List[Element]]) -> Element:
    """Create new element

    Args:
        bounding: Bounding coordinates [SW, SE, NE, NW]
        parent: Parent element
        u_level: Refinement level
        max_lv: Maximum refinement level
        vvp_level_element_list: Level-wise element list

    Returns:
        New Element
    """
    # Ensure level list has capacity
    while len(vvp_level_element_list) <= u_level:
        vvp_level_element_list.append([])

    # Create new element with corner coordinates
    element = Element(pParent=parent, uLevel=u_level)

    # Set corner coordinates
    for i in range(4):
        element.acNodalCoordinates[i] = bounding[i]

    # Add to level list
    vvp_level_element_list[u_level].append(element)

    # Set self reference
    element.pMe = element

    return element


def delete_unused_elements(vvp_level_element_list: List[List[Element]]) -> None:
    """Delete unused elements from the list

    Args:
        vvp_level_element_list: Level-wise element list
    """
    new_list = [[] for _ in range(len(vvp_level_element_list))]

    for lv in range(len(vvp_level_element_list)):
        for e in range(len(vvp_level_element_list[lv])):
            elem = vvp_level_element_list[lv][e]
            # Keep elements with valid self-reference
            if elem.pMe is not None and lv < len(vvp_level_element_list):
                new_list[lv].append(elem)

    # Update the list in place
    for lv in range(len(vvp_level_element_list)):
        vvp_level_element_list[lv] = new_list[lv]


def quadtree_initialization(nx: float, ny: float, max_lv: int, gamma: float,
                            vvp_level_element_list: List[List[Element]],
                            m_node_coordinate_list: Dict[Coord, int],
                            vv_eft: List[List[int]], option: int,
                            m_phi_coordinate_list: Dict[Coord, float],
                            m_u_coordinate_list: Dict[Coord, float],
                            m_phi_velocity_coordinate_list: Dict[Coord, float],
                            m_u_velocity_coordinate_list: Dict[Coord, float]) -> None:
    """Initialize quadtree mesh

    Args:
        nx: Domain width
        ny: Domain height
        max_lv: Maximum refinement levels
        gamma: Error tolerance
        vvp_level_element_list: Level-wise element list (output)
        m_node_coordinate_list: Node coordinate map (output)
        vv_eft: Element freedom table (output)
        option: Refinement criterion
        m_phi_coordinate_list: Phase field values (output)
        m_u_coordinate_list: Temperature/displacement values (output)
        m_phi_velocity_coordinate_list: Phase field velocity (output)
        m_u_velocity_coordinate_list: Temperature/displacement velocity (output)
    """
    # Resize level element list
    vvp_level_element_list.clear()
    vvp_level_element_list.extend([] for _ in range(max_lv + 1))

    # Construct root
    quadtree_construct_root(nx, ny, max_lv, vvp_level_element_list)

    # Initialize coordinate maps with initial values based on root element corners
    if vvp_level_element_list and vvp_level_element_list[0]:
        root = vvp_level_element_list[0][0]
        for i in range(4):
            coord = root.acNodalCoordinates[i]
            # Initialize with default values based on position
            if option == 11:  # Initialization - phase field from interface
                rad = nx / 164.0
                dist = coord.y - rad
                m_phi_coordinate_list[coord] = -np.tanh(dist / np.sqrt(2))
            else:
                m_phi_coordinate_list[coord] = 0.0
            m_u_coordinate_list[coord] = 0.0
            m_phi_velocity_coordinate_list[coord] = 0.0
            m_u_velocity_coordinate_list[coord] = 0.0

    # Initial subdivision
    quadtree_subdivide(max_lv, gamma, vvp_level_element_list, option,
                       m_phi_coordinate_list, m_u_coordinate_list,
                       m_phi_velocity_coordinate_list, m_u_velocity_coordinate_list)


def quadtree_mesh_generate(max_lv: int, gamma: float,
                           vvp_level_element_list: List[List[Element]],
                           option: int,
                           m_phi_coordinate_list: Dict[Coord, float],
                           m_u_coordinate_list: Dict[Coord, float],
                           m_phi_velocity_coordinate_list: Dict[Coord, float],
                           m_u_velocity_coordinate_list: Dict[Coord, float]) -> None:
    """Generate quadtree mesh through fuse-subdivide cycle

    Args:
        max_lv: Maximum refinement levels
        gamma: Error tolerance
        vvp_level_element_list: Level-wise element list
        option: Refinement criterion
        m_phi_coordinate_list: Phase field values
        m_u_coordinate_list: Temperature/displacement values
        m_phi_velocity_coordinate_list: Phase field velocity
        m_u_velocity_coordinate_list: Temperature/displacement velocity
    """
    # Fuse from max_lv-1 to 0
    quadtree_fuse(max_lv, gamma, vvp_level_element_list, option,
                  m_phi_coordinate_list, m_u_coordinate_list,
                  m_phi_velocity_coordinate_list, m_u_velocity_coordinate_list)

    # Subdivide from 0 to max_lv
    quadtree_subdivide(max_lv, gamma, vvp_level_element_list, option,
                       m_phi_coordinate_list, m_u_coordinate_list,
                       m_phi_velocity_coordinate_list, m_u_velocity_coordinate_list)


def quadtree_construct_root(nx: float, ny: float, max_lv: int,
                            vvp_level_element_list: List[List[Element]]) -> None:
    """Construct root element

    Args:
        nx: Domain width
        ny: Domain height
        max_lv: Maximum refinement levels
        vvp_level_element_list: Level-wise element list (output)
    """
    sw = Coord(0, 0)
    se = Coord(nx, 0)
    ne = Coord(nx, ny)
    nw = Coord(0, ny)  # Note: using ny, not nx (fixed from C++ bug)
    root_coordinates = [sw, se, ne, nw]

    create_element(root_coordinates, None, 0, max_lv, vvp_level_element_list)


def quadtree_subdivide(max_lv: int, gamma: float,
                       vvp_level_element_list: List[List[Element]],
                       option: int,
                       m_phi_coordinate_list: Dict[Coord, float],
                       m_u_coordinate_list: Dict[Coord, float],
                       m_phi_velocity_coordinate_list: Dict[Coord, float],
                       m_u_velocity_coordinate_list: Dict[Coord, float]) -> None:
    """Subdivide elements based on error criteria

    Args:
        max_lv: Maximum refinement levels
        gamma: Error tolerance
        vvp_level_element_list: Level-wise element list
        option: Refinement criterion
        m_phi_coordinate_list: Phase field values
        m_u_coordinate_list: Temperature/displacement values
        m_phi_velocity_coordinate_list: Phase field velocity
        m_u_velocity_coordinate_list: Temperature/displacement velocity
    """
    for lv in range(max_lv):
        if lv < len(vvp_level_element_list):
            for i in range(len(vvp_level_element_list[lv])):
                elem = vvp_level_element_list[lv][i]
                if elem.check_error(option, m_phi_coordinate_list, m_u_coordinate_list, gamma):
                    elem.subdivide(
                        max_lv, vvp_level_element_list,
                        m_phi_coordinate_list, m_u_coordinate_list,
                        m_phi_velocity_coordinate_list, m_u_velocity_coordinate_list)


def quadtree_fuse(max_lv: int, gamma: float,
                 vvp_level_element_list: List[List[Element]],
                 option: int,
                 m_phi_coordinate_list: Dict[Coord, float],
                 m_u_coordinate_list: Dict[Coord, float],
                 m_phi_velocity_coordinate_list: Dict[Coord, float],
                 m_u_velocity_coordinate_list: Dict[Coord, float]) -> None:
    """Fuse elements if error is small enough

    Args:
        max_lv: Maximum refinement levels
        gamma: Error tolerance
        vvp_level_element_list: Level-wise element list
        option: Refinement criterion
        m_phi_coordinate_list: Phase field values
        m_u_coordinate_list: Temperature/displacement values
        m_phi_velocity_coordinate_list: Phase field velocity
        m_u_velocity_coordinate_list: Temperature/displacement velocity
    """
    for lv in range(max_lv):
        reverse_lv = max_lv - 1 - lv
        if reverse_lv < len(vvp_level_element_list):
            for i in range(len(vvp_level_element_list[reverse_lv])):
                elem = vvp_level_element_list[reverse_lv][i]
                elem.fuse(
                    vvp_level_element_list, option,
                    m_phi_coordinate_list, m_u_coordinate_list,
                    m_phi_velocity_coordinate_list, m_u_velocity_coordinate_list,
                    gamma)

    # Delete unused elements
    delete_unused_elements(vvp_level_element_list)


def quadtree_add_nodes(vvp_level_element_list: List[List[Element]],
                       m_node_coordinate_list: Dict[Coord, int]) -> None:
    """Add nodes to all leaf elements

    Args:
        vvp_level_element_list: Level-wise element list
        m_node_coordinate_list: Node coordinate map (output)
    """
    # Clear existing nodes
    m_node_coordinate_list.clear()

    # Add nodes from leaf elements
    for lv in range(len(vvp_level_element_list)):
        for e in range(len(vvp_level_element_list[lv])):
            elem = vvp_level_element_list[lv][e]
            if elem.apChildren[0] is None:  # Leaf element
                elem.add_nodes(m_node_coordinate_list)


def report_element(vvp_level_element_list: List[List[Element]],
                   vp_final_element_list: List[Element],
                   m_node_coordinate_list: Dict[Coord, int],
                   vv_eft: List[List[int]],
                   vc_node_coordinates: List[Coord]) -> None:
    """Generate final element report

    Args:
        vvp_level_element_list: Level-wise element list
        vp_final_element_list: Final leaf elements (output)
        m_node_coordinate_list: Node coordinate map
        vv_eft: Element freedom table (output)
        vc_node_coordinates: Node coordinates (output)
    """
    # Clear outputs
    vp_final_element_list.clear()
    vv_eft.clear()
    vc_node_coordinates.clear()

    # Collect leaf elements within domain
    for lv in range(len(vvp_level_element_list)):
        for e in range(len(vvp_level_element_list[lv])):
            elem = vvp_level_element_list[lv][e]
            if (elem.apChildren[0] is None and
                elem.acNodalCoordinates[0].x >= 0 and
                elem.acNodalCoordinates[1].x <= 204.8):
                vp_final_element_list.append(elem)

    # Build node to coordinate map
    m_node2coord: Dict[int, Coord] = {}
    for coord, node_id in m_node_coordinate_list.items():
        m_node2coord[node_id] = coord

    # Sort nodes by ID and collect coordinates
    for node_id in sorted(m_node2coord.keys()):
        vc_node_coordinates.append(m_node2coord[node_id])

    # Build element freedom table
    for elem in vp_final_element_list:
        element_nodes = []
        for i in range(8):
            if (elem.bitElementType >> i) & 1:
                coord = elem.acNodalCoordinates[i]
                if coord in m_node_coordinate_list:
                    element_nodes.append(m_node_coordinate_list[coord])
        vv_eft.append(element_nodes)


# Convenience function for easy mesh generation
def create_quadtree_mesh(nx: float = 204.8, ny: float = 204.8,
                         max_levels: int = 6, gamma: float = 0.1,
                         option: int = 12) -> Tuple[np.ndarray, List[List[int]]]:
    """Create a quadtree mesh and return node coordinates and EFT

    Args:
        nx: Domain width
        ny: Domain height
        max_levels: Maximum refinement levels
        gamma: Error tolerance
        option: Refinement criterion

    Returns:
        Tuple of (node_coordinates, element_freedom_table)
    """
    mesh = QuadtreeMesh(nx, ny, max_levels, gamma)
    mesh.generate(option)
    return mesh.get_node_coordinates(), mesh.vv_eft


if __name__ == "__main__":
    # Test the quadtree implementation
    print("Testing Quadtree AMR implementation...")

    # Create simple mesh
    mesh = QuadtreeMesh(nx=10.0, ny=10.0, max_levels=3, gamma=0.1)

    # Initialize with regular mesh option
    mesh.generate(option=12)

    print(f"Nodes: {mesh.get_node_count()}")
    print(f"Elements: {mesh.get_element_count()}")

    # Get node coordinates
    coords = mesh.get_node_coordinates()
    print(f"Node coordinate shape: {coords.shape}")
    print(f"First few nodes:\n{coords[:5]}")

    print("Test passed!")
