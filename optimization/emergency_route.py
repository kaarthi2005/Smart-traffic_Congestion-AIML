import networkx as nx
import xml.etree.ElementTree as ET
import pandas as pd


NETWORK_FILE = "simulation/network.net.xml"
TRAFFIC_FILE = "data/traffic.csv"


# ---------------------------------------
# 1. Load SUMO road network
# ---------------------------------------

def load_network():

    graph = nx.DiGraph()

    tree = ET.parse(NETWORK_FILE)
    root = tree.getroot()

    for edge in root.findall("edge"):

        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")

        # Ignore internal SUMO edges
        if from_node is None or to_node is None:
            continue

        lane = edge.find("lane")

        if lane is not None:
            length = float(lane.get("length", 1))
        else:
            length = 1

        graph.add_edge(
            from_node,
            to_node,
            weight=length,
            length=length,
            edge_id=edge_id
        )

    return graph


# ---------------------------------------
# 2. Load traffic data
# ---------------------------------------

def load_traffic():

    df = pd.read_csv(TRAFFIC_FILE)

    traffic = (
        df.groupby("edge")
        .agg(
            average_speed=("speed", "mean"),
            vehicle_count=("vehicle_id", "nunique")
        )
        .reset_index()
    )

    return traffic


# ---------------------------------------
# 3. Calculate traffic penalty
# ---------------------------------------

def calculate_penalty(speed, vehicle_count):

    # Higher vehicles = higher cost
    # Lower speed = higher cost

    if speed < 5:
        return 5

    elif speed < 8:
        return 4

    elif speed < 11:
        return 3

    elif speed < 14:
        return 2

    else:
        return 1


# ---------------------------------------
# 4. Add traffic cost to network
# ---------------------------------------

def apply_traffic_cost(graph, traffic):

    traffic_map = {}

    for _, row in traffic.iterrows():

        traffic_map[row["edge"]] = {
            "speed": row["average_speed"],
            "vehicles": row["vehicle_count"]
        }

    for u, v, data in graph.edges(data=True):

        edge_id = data["edge_id"]

        length = data["length"]

        if edge_id in traffic_map:

            speed = traffic_map[edge_id]["speed"]

            vehicles = traffic_map[edge_id]["vehicles"]

            penalty = calculate_penalty(
                speed,
                vehicles
            )

        else:

            penalty = 1

            speed = 13.89

            vehicles = 0

        # Traffic-aware cost
        traffic_cost = length * penalty

        data["weight"] = traffic_cost

        data["average_speed"] = speed

        data["vehicles"] = vehicles

        data["penalty"] = penalty


# ---------------------------------------
# 5. Find emergency route
# ---------------------------------------

def find_emergency_route(
    graph,
    source,
    destination
):

    route = nx.shortest_path(
        graph,
        source=source,
        target=destination,
        weight="weight"
    )

    return route


# ---------------------------------------
# MAIN
# ---------------------------------------

print("Loading road network...")

graph = load_network()

print(
    "Network loaded!",
    graph.number_of_nodes(),
    "nodes",
    graph.number_of_edges(),
    "roads"
)


print("\nLoading traffic data...")

traffic = load_traffic()

print(
    "Traffic records:",
    len(traffic)
)


print("\nApplying traffic costs...")

apply_traffic_cost(
    graph,
    traffic
)

print("Traffic costs applied!")


# ---------------------------------------
# Select source and destination
# ---------------------------------------

nodes = list(graph.nodes)

source = nodes[0]

destination = nodes[-1]


print("\nEmergency vehicle:")
print("Source:", source)
print("Destination:", destination)


# ---------------------------------------
# Calculate route
# ---------------------------------------

try:

    route = find_emergency_route(
        graph,
        source,
        destination
    )

    print("\n================================")
    print("TRAFFIC-AWARE EMERGENCY ROUTE")
    print("================================")

    print(
        " → ".join(route)
    )


    # Calculate total route cost

    total_cost = 0

    print("\nRoad details:")

    for i in range(len(route) - 1):

        u = route[i]

        v = route[i + 1]

        data = graph[u][v]

        total_cost += data["weight"]

        print(
            f"{data['edge_id']} | "
            f"Speed: {data['average_speed']:.2f} | "
            f"Vehicles: {data['vehicles']} | "
            f"Penalty: {data['penalty']} | "
            f"Cost: {data['weight']:.2f}"
        )


    print(
        "\nTotal route cost:",
        round(total_cost, 2)
    )


except nx.NetworkXNoPath:

    print(
        "No route exists between",
        source,
        "and",
        destination
    )