from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypeAlias

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

Q: TypeAlias = NDArray[np.float64]


@dataclass(eq=False)
class GraphNode:
    parent: Optional[GraphNode]
    transition: list[Q]  # grasping configurations from the parent's q (excluded) to q (included)
    transition_placements: list[pin.SE3]  # cube placement for each q in transition
    q: Q = field(init=False)
    cube_placement: pin.SE3 = field(init=False)

    def __post_init__(self) -> None:
        if not self.transition or not self.transition_placements:
            raise ValueError("GraphNode needs non-empty transitions and transition placements")
        self.q = self.transition[-1]
        self.cube_placement = self.transition_placements[-1]


@dataclass(eq=False)
class KDTreeNode:
    graph_node: GraphNode
    axis: int
    left: Optional[KDTreeNode] = None
    right: Optional[KDTreeNode] = None


class RRTTree:
    def __init__(self, root: GraphNode):
        self.nodes = [root]
        self.kd_root = KDTreeNode(root, axis=0)

    def __len__(self) -> int:
        return len(self.nodes)

    def add_edge_and_vertex(self, parent: GraphNode, transition: list[Q], placements: list[pin.SE3]) -> GraphNode:
        node = GraphNode(parent=parent, transition=transition, transition_placements=placements)
        self.nodes.append(node)
        self._insert_kd(self.kd_root, node, depth=0)
        return node

    def nearest_vertex(self, cube_target: pin.SE3) -> GraphNode:
        _, node = self._nearest_kd(self.kd_root, cube_target.translation, best=(float("inf"), self.kd_root.graph_node))
        return node

    def getpath(self, node: GraphNode) -> tuple[list[Q], list[pin.SE3]]:
        qs, placements = [], []
        while node.parent is not None:
            qs = node.transition + qs
            placements = node.transition_placements + placements
            node = node.parent
        return [node.q] + qs, [node.cube_placement] + placements

    def _insert_kd(self, kd_node: KDTreeNode, graph_node: GraphNode, depth: int) -> None:
        axis = kd_node.axis
        if graph_node.cube_placement.translation[axis] < kd_node.graph_node.cube_placement.translation[axis]:
            if kd_node.left is None:
                kd_node.left = KDTreeNode(graph_node, axis=(depth + 1) % 3)
            else:
                self._insert_kd(kd_node.left, graph_node, depth + 1)
        else:
            if kd_node.right is None:
                kd_node.right = KDTreeNode(graph_node, axis=(depth + 1) % 3)
            else:
                self._insert_kd(kd_node.right, graph_node, depth + 1)

    def _nearest_kd(self, kd_node: Optional[KDTreeNode], target: Q, best: tuple[float, GraphNode],) -> tuple[float, GraphNode]:
        if kd_node is None:
            return best

        point = kd_node.graph_node.cube_placement.translation
        dist = _squared_distance(point, target)
        if dist < best[0]:
            best = (dist, kd_node.graph_node)

        axis = kd_node.axis
        delta = target[axis] - point[axis]

        if delta < 0:
            near_branch, far_branch = kd_node.left, kd_node.right
        else:
            near_branch, far_branch = kd_node.right, kd_node.left

        best = self._nearest_kd(near_branch, target, best)
        if delta * delta < best[0]:
            best = self._nearest_kd(far_branch, target, best)

        return best


def _squared_distance(p1: Q, p2: Q) -> float:
    diff = p1 - p2
    return float(diff @ diff)


if __name__ == "__main__":
    def placement(x: float, y: float, z: float) -> pin.SE3:
        return pin.SE3(np.eye(3), np.array([x, y, z]))

    def q(value: float) -> Q:
        return np.array([value])

    root = GraphNode(parent=None, transition=[q(0.0)], transition_placements=[placement(0.0, 0.0, 0.0)])
    tree = RRTTree(root)

    node_a = tree.add_edge_and_vertex(
        parent=root,
        transition=[q(0.5), q(1.0)],
        placements=[placement(0.5, 0.0, 0.0), placement(1.0, 0.0, 0.0)],
    )
    node_b = tree.add_edge_and_vertex(
        parent=node_a,
        transition=[q(1.5), q(2.0)],
        placements=[placement(1.5, 1.0, 0.0), placement(2.0, 1.0, 0.0)],
    )
    tree.add_edge_and_vertex(
        parent=root,
        transition=[q(3.0)],
        placements=[placement(0.0, 3.0, 0.0)],
    )

    target = placement(1.8, 1.1, 0.0)
    nearest = tree.nearest_vertex(target)
    path_qs, path_placements = tree.getpath(node_b)

    print(f"tree size: {len(tree)}")
    print(f"target cube position: {target.translation}")
    print(f"nearest cube position: {nearest.cube_placement.translation}")
    print("path to node_b:")
    for q_i, placement_i in zip(path_qs, path_placements):
        print(f"q={q_i}, cube={placement_i.translation}")
