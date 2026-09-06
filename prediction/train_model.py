import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load traffic features
df = pd.read_csv("data/traffic_features.csv")

print("Dataset loaded!")
print("Records:", len(df))

# Create previous-time features
df["previous_speed"] = df["average_speed"].shift(1)
df["previous_vehicles"] = df["vehicle_count"].shift(1)

# Remove first row because it has no previous value
df = df.dropna()

# Input features
X = df[
    [
        "previous_speed",
        "previous_vehicles"
    ]
]

# Target
y = df["average_speed"]

# Split data chronologically
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

print("\nModel results:")
print("MAE:", mae)
print("RMSE:", rmse)

# Save model
joblib.dump(
    model,
    "prediction/traffic_model.pkl"
)

print("\nModel saved to:")
print("prediction/traffic_model.pkl")