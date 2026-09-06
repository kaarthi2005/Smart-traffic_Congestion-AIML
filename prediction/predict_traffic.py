import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


DATA_FILE = "data/sumo_traffic_features.csv"
MODEL_FILE = "prediction/traffic_lstm_sumo.keras"
SCALER_FILE = "prediction/scaler_sumo.pkl"


def predict_traffic():

    print("Loading traffic data...")

    df = pd.read_csv(DATA_FILE)

    features = [
        "average_speed",
        "vehicle_count"
    ]

    data = df[features].values

    scaler = joblib.load(SCALER_FILE)

    scaled_data = scaler.transform(data)

    sequence = scaled_data[-10:]

    X = np.array([sequence])

    model = load_model(MODEL_FILE, compile=False)

    prediction = model.predict(X, verbose=0)

    predicted_scaled = prediction[0][0]

    dummy = np.zeros((1, 2))
    dummy[0][0] = predicted_scaled

    predicted = scaler.inverse_transform(dummy)

    predicted_speed = float(predicted[0][0])

    current_speed = float(df["average_speed"].iloc[-1])

    if predicted_speed < 8:
        congestion = "HIGH"

    elif predicted_speed < 11:
        congestion = "MEDIUM"

    else:
        congestion = "LOW"

    return {
        "current_speed": current_speed,
        "predicted_speed": predicted_speed,
        "congestion": congestion
    }


if __name__ == "__main__":

    result = predict_traffic()

    print("\n==============================")
    print("LSTM TRAFFIC PREDICTION")
    print("==============================")

    print(
        "Current average speed:",
        round(result["current_speed"], 2)
    )

    print(
        "Predicted average speed:",
        round(result["predicted_speed"], 2)
    )

    print(
        "Predicted congestion:",
        result["congestion"]
    )

    print("==============================")