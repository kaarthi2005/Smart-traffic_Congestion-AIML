import pandas as pd
import numpy as np

# Load traffic data
df = pd.read_csv("data/traffic.csv")

print("Traffic data loaded!")
print("Total records:", len(df))
print("Total vehicles:", df["vehicle_id"].nunique())
print("Total roads:", df["edge"].nunique())

# Average speed for each road
road_stats = df.groupby("edge").agg(
    vehicle_count=("vehicle_id", "nunique"),
    average_speed=("speed", "mean"),
    min_speed=("speed", "min"),
    max_speed=("speed", "max")
).reset_index()

# Define congestion level
def congestion_level(speed):
    if speed < 5:
        return "HIGH"
    elif speed < 10:
        return "MEDIUM"
    else:
        return "LOW"

road_stats["congestion"] = road_stats["average_speed"].apply(
    congestion_level
)

# Sort most congested roads first
road_stats = road_stats.sort_values("average_speed")

print("\nRoad congestion:")
print(road_stats.to_string(index=False))

# Save results
road_stats.to_csv("data/congestion.csv", index=False)

print("\nCongestion data saved to data/congestion.csv")