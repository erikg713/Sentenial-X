from sentenial_x.plugins import PluginBase
from networkx import DiGraph, shortest_path_length

class AttackGraphPlugin(PluginBase):
    """
    Ingest assets + topology, build attacker-path graph, emit prioritized choke points.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = DiGraph()

    def on_inventory_update(self, event):
        assets = event.payload  # list of asset dicts
        for asset in assets:
            self.graph.add_node(asset['id'], **asset)

    def on_topology_update(self, event):
        links = event.payload  # list of (src, dst, attributes)
        for src, dst, attrs in links:
            self.graph.add_edge(src, dst, **attrs)

    def on_scheduled_task(self, event):
        # compute shortest paths from internet-facing nodes to critical assets
        critical = [n for n, data in self.graph.nodes(data=True) if data.get('critical')]
        internet_facing = [n for n, data in self.graph.nodes(data=True) if data.get('exposed')]
        choke_scores = {}
        for i in internet_facing:
            for c in critical:
                try:
                    dist = shortest_path_length(self.graph, source=i, target=c)
                    choke_scores[(i, c)] = dist
                except Exception:
                    continue

        # pick top 5 shortest paths as highest risk
        prioritized = sorted(choke_scores.items(), key=lambda x: x[1])[:5]
        self.emit("on_attack_paths_computed", {"paths": prioritized})
