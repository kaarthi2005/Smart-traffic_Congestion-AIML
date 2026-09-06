import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/sumo_traffic_features.csv")

print("Traffic dataset loaded!")
print("Total records:", len(df))


# ==========================================
# 2. SELECT FEATURES
# ==========================================

data = df[
    ["average_speed", "vehicle_count"]
].values


# ==========================================
# 3. NORMALIZE DATA
# ==========================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)


# ==========================================
# 4. CREATE TIME SEQUENCES
# ==========================================

sequence_length = 10

X = []
y = []

for i in range(sequence_length, len(scaled_data)):

    X.append(
        scaled_data[i-sequence_length:i]
    )

    # Predict next average speed
    y.append(
        scaled_data[i, 0]
    )


X = np.array(X)
y = np.array(y)


print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. BUILD LSTM
# ==========================================

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(32)
)

model.add(Dropout(0.2))

model.add(
    Dense(16, activation="relu")
)

model.add(
    Dense(1)
)


# ==========================================
# 7. COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


model.summary()


# ==========================================
# 8. TRAIN
# ==========================================

print("\nTraining LSTM...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)


# ==========================================
# 9. PREDICT
# ==========================================

predicted_scaled = model.predict(
    X_test
)


# ==========================================
# 10. CONVERT BACK TO SPEED
# ==========================================

# Reconstruct two columns so scaler can
# inverse-transform the predicted speed.

dummy = np.zeros(
    (len(predicted_scaled), 2)
)

dummy[:, 0] = predicted_scaled[:, 0]

predicted_speed = scaler.inverse_transform(
    dummy
)[:, 0]


dummy_actual = np.zeros(
    (len(y_test), 2)
)

dummy_actual[:, 0] = y_test

actual_speed = scaler.inverse_transform(
    dummy_actual
)[:, 0]


# ==========================================
# 11. EVALUATE
# ==========================================

mae = mean_absolute_error(
    actual_speed,
    predicted_speed
)

rmse = np.sqrt(
    mean_squared_error(
        actual_speed,
        predicted_speed
    )
)


print("\n==============================")
print("LSTM RESULTS")
print("==============================")

print("MAE :", mae)
print("RMSE:", rmse)


# ==========================================
# 12. SAVE MODEL
# ==========================================

model.save("prediction/traffic_lstm_sumo.keras")

joblib.dump(scaler, "prediction/scaler_sumo.pkl")


print("\nModel saved:")
print("prediction/traffic_lstm.keras")

print("\nScaler saved:")
print("prediction/scaler.pkl")