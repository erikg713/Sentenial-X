# chain-executor/graph_utils.py

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class Node:
    """Represents a node in the execution graph."""

    def __init__(self, node_id: str, payload: Optional[Dict[str, Any]] = None):
        self.id = node_id
        self.payload = payload or {}
        self.edges: List[str] = []  # connected node IDs

    def add_edge(self, target_id: str) -> None:
        if target_id not in self.edges:
            self.edges.append(target_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "payload": self.payload,
            "edges": self.edges,
        }


class ExecutionGraph:
    """Graph structure for ordered task execution with dependency tracking."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node_id: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, payload)
            logger.debug(f"Node added: {node_id}")
        else:
            logger.warning(f"Node '{node_id}' already exists. Skipping add.")

    def add_edge(self, source_id: str, target_id: str) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Cannot add edge, one or both nodes missing: {source_id}, {target_id}")
        self.nodes[source_id].add_edge(target_id)
        logger.debug(f"Edge added: {source_id} -> {target_id}")

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def to_dict(self) -> Dict[str, Any]:
        return {node_id: node.to_dict() for node_id, node in self.nodes.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self.nodes.clear()
        for node_id, node_data in data.items():
            node = Node(node_id, node_data.get("payload"))
            node.edges = node_data.get("edges", [])
            self.nodes[node_id] = node
        logger.info("ExecutionGraph loaded from dict")

    def detect_cycles(self) -> bool:
        """Detect cycles using DFS."""
        visited: Set[str] = set()
        recursion_stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            recursion_stack.add(node_id)

            for neighbor in self.nodes[node_id].edges:
                if neighbor not in visited and dfs(neighbor):
                    return True
                elif neighbor in recursion_stack:
                    return True

            recursion_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited and dfs(node_id):
                logger.error(f"Cycle detected starting at node: {node_id}")
                return True
        return False

    def topological_sort(self) -> List[str]:
        """Return a list of nodes in topological order if DAG is valid."""
        visited: Set[str] = set()
        stack: List[str] = []

        def dfs(node_id: str) -> None:
            visited.add(node_id)
            for neighbor in self.nodes[node_id].edges:
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)

        stack.reverse()
        logger.debug(f"Topological sort result: {stack}")
        return stack

    def validate(self) -> Tuple[bool, str]:
        """Validate the graph for execution readiness."""
        if not self.nodes:
            return False, "Graph is empty"

        if self.detect_cycles():
            return False, "Graph contains cycles"

        try:
            self.topological_sort()
        except Exception as e:
            return False, f"Validation failed during sorting: {e}"

        return True, "Graph is valid"


# Utility function for quick graph construction
def build_execution_graph(edges: List[Tuple[str, str]], payloads: Optional[Dict[str, Dict[str, Any]]] = None) -> ExecutionGraph:
    """
    Build an ExecutionGraph from a list of edges and optional payloads.
    
    Example:
        edges = [("A", "B"), ("B", "C")]
        payloads = {"A": {"task": "start"}, "B": {"task": "process"}, "C": {"task": "end"}}
    """
    graph = ExecutionGraph()
    payloads = payloads or {}

    # Add nodes
    all_nodes = {src for src, _ in edges} | {dst for _, dst in edges}
    for node_id in all_nodes:
        graph.add_node(node_id, payloads.get(node_id))

    # Add edges
    for src, dst in edges:
        graph.add_edge(src, dst)

    return graph
