import os
import pandas as pd


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sumo_traffic.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sumo_traffic_features.csv"
)


# --------------------------------------------------
# LOAD SUMO DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} vehicle records")


# --------------------------------------------------
# REMOVE INTERNAL SUMO ROADS
# --------------------------------------------------

df = df[
    ~df["road"].astype(str).str.startswith(":")
].copy()


# --------------------------------------------------
# CALCULATE TRAFFIC FEATURES
# --------------------------------------------------

features = (
    df.groupby("time")
    .agg(
        average_speed=("speed", "mean"),
        vehicle_count=("vehicle_id", "nunique")
    )
    .reset_index()
)


# --------------------------------------------------
# CALCULATE CONGESTION
# --------------------------------------------------

def calculate_congestion(speed):

    if speed >= 12:
        return "LOW"

    elif speed >= 6:
        return "MEDIUM"

    else:
        return "HIGH"


features["congestion"] = features[
    "average_speed"
].apply(calculate_congestion)


# --------------------------------------------------
# ROUND VALUES
# --------------------------------------------------

features["average_speed"] = features[
    "average_speed"
].round(2)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print("\nTraffic Features:")
print(features.head(10))

print(
    f"\n Saved to: {OUTPUT_FILE}"
)

print(
    f"Total time steps: {len(features)}"
)