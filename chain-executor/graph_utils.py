import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("GraphUtils")
logging.basicConfig(level=logging.INFO)

class ExecutionGraphError(Exception):
    """Custom error for graph validation or execution issues."""
    pass

def build_adjacency_list(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Builds an adjacency list from edge list (source → target)."""
    graph: Dict[str, List[str]] = {}
    for src, tgt in edges:
        graph.setdefault(src, []).append(tgt)
        graph.setdefault(tgt, [])  # ensure all nodes appear
    return graph

def detect_cycles(graph: Dict[str, List[str]]) -> bool:
    """Detects if the graph contains a cycle using DFS."""
    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """Performs a topological sort (Kahn's algorithm)."""
    in_degree: Dict[str, int] = {node: 0 for node in graph}
    for neighbors in graph.values():
        for neighbor in neighbors:
            in_degree[neighbor] += 1

    queue = [node for node in graph if in_degree[node] == 0]
    sorted_order = []

    while queue:
        current = queue.pop(0)
        sorted_order.append(current)
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_order) != len(graph):
        raise ExecutionGraphError("Graph has a cycle or unresolved dependencies.")

    return sorted_order

def extract_nodes_and_edges(steps: List[Dict]) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Extracts nodes and edges from structured step metadata."""
    edges: List[Tuple[str, str]] = []
    nodes: Set[str] = set()

    for step in steps:
        name = step["name"]
        nodes.add(name)
        for dep in step.get("depends_on", []):
            edges.append((dep, name))  # dep → current

    return nodes, edges

