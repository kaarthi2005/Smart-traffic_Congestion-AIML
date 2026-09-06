"""Compare static, traffic-aware, and emergency routing strategies."""
import argparse
import os
import xml.etree.ElementTree as ET
import networkx as nx
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_FILE = os.path.join(BASE_DIR, "simulation", "network.net.xml")
TRAFFIC_FILE = os.path.join(BASE_DIR, "data", "sumo_traffic.csv")


def load_graph():
    graph = nx.DiGraph()
    root = ET.parse(NETWORK_FILE).getroot()
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        lane = edge.find("lane")
        if lane is None:
            continue
        u, v = edge.get("from"), edge.get("to")
        if not u or not v:
            continue
        graph.add_edge(u, v, edge_id=edge_id,
                       length=float(lane.get("length", 1.0)),
                       max_speed=float(lane.get("speed", 13.89)))
    return graph


def apply_traffic_costs(graph):
    if not os.path.exists(TRAFFIC_FILE):
        for _, _, d in graph.edges(data=True):
            d.update(speed=d["max_speed"], vehicles=0, cost=d["length"] / max(d["max_speed"], 0.1))
        return
    df = pd.read_csv(TRAFFIC_FILE)
    road_col = "road" if "road" in df.columns else "edge"
    grouped = df.groupby(road_col).agg(speed=("speed", "mean"), vehicles=("vehicle_id", "nunique"))
    for _, _, d in graph.edges(data=True):
        row = grouped.loc[d["edge_id"]] if d["edge_id"] in grouped.index else None
        speed = float(row["speed"]) if row is not None else d["max_speed"]
        vehicles = int(row["vehicles"]) if row is not None else 0
        speed = max(speed, 0.1)
        congestion_penalty = 1.0 + min(2.0, vehicles / 20.0)
        d.update(speed=speed, vehicles=vehicles,
                 cost=(d["length"] / speed) * congestion_penalty)


def edge_ids(graph, nodes):
    return [graph[nodes[i]][nodes[i + 1]]["edge_id"] for i in range(len(nodes) - 1)]


def explain(static, traffic, graph):
    static_edges = set(edge_ids(graph, static))
    traffic_edges = set(edge_ids(graph, traffic))
    changed = [e for e in traffic_edges if e not in static_edges]
    if not changed:
        return "The traffic-aware route matches the static shortest route because current traffic does not justify a detour."
    details=[]
    for e in changed:
        for _, _, d in graph.edges(data=True):
            if d["edge_id"] == e:
                details.append(f"{e} has about {d['vehicles']} vehicles and {d['speed']:.1f} m/s average speed")
                break
    return "AI prefers a traffic-aware alternative to reduce estimated delay. " + "; ".join(details) + "."


def compare(source="A0", destination="C2"):
    graph=load_graph(); apply_traffic_costs(graph)
    static=nx.shortest_path(graph, source, destination, weight="length")
    traffic=nx.shortest_path(graph, source, destination, weight="cost")
    return {
        "static": {"nodes": static, "edges": edge_ids(graph, static), "cost": sum(graph[static[i]][static[i+1]]["length"] for i in range(len(static)-1))},
        "traffic_aware": {"nodes": traffic, "edges": edge_ids(graph, traffic), "cost": sum(graph[traffic[i]][traffic[i+1]]["cost"] for i in range(len(traffic)-1))},
        "explanation": explain(static, traffic, graph),
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",default="A0"); p.add_argument("--destination",default="C2"); args=p.parse_args()
    result=compare(args.source,args.destination)
    print("ROUTE COMPARISON")
    print("Static shortest :", " -> ".join(result["static"]["nodes"]))
    print("Traffic-aware   :", " -> ".join(result["traffic_aware"]["nodes"]))
    print("Explanation     :", result["explanation"])

if __name__ == "__main__":
    main()
