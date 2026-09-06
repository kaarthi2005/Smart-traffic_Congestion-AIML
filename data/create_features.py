import pandas as pd

# Load raw traffic data
df = pd.read_csv("data/traffic.csv")

print("Raw records:", len(df))

# Group traffic by simulation time
features = df.groupby("time").agg(
    average_speed=("speed", "mean"),
    vehicle_count=("vehicle_id", "nunique")
).reset_index()

# Calculate congestion
def congestion_level(speed):
    if speed < 5:
        return "HIGH"
    elif speed < 10:
        return "MEDIUM"
    else:
        return "LOW"

features["congestion"] = features["average_speed"].apply(
    congestion_level
)

# Save ML-ready dataset
features.to_csv(
    "data/traffic_features.csv",
    index=False
)

print("\nFeature dataset created!")
print("Records:", len(features))

print("\nFirst 10 rows:")
print(features.head(10).to_string(index=False))

print("\nSaved to:")
print("data/traffic_features.csv")