import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from RegressionTree import RegressionTree
import time

#Task 4.1
def get_future_states(x1, x2):
    x1_future_state = 0.9 * x1 - 0.2 * x2
    x2_future_state = 0.2 * x1 + 0.9 * x2
    return x1_future_state, x2_future_state

def generate_samples(x1, x2, time):
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    for t in range(0, time):
        x1_list.append(x1)
        x2_list.append(x2)
        x1, x2 = get_future_states(x1, x2)
        y1_list.append(x1)
        y2_list.append(x2)
    data = np.hstack((
        np.array(x1_list).reshape(-1, 1),
        np.array(x2_list).reshape(-1, 1),
        np.array(y1_list).reshape(-1, 1),
        np.array(y2_list).reshape(-1, 1)
    ))
    df = pd.DataFrame(data)
    df.to_csv("q4_part1_dataset.csv", index=False, header=False)
    return df

# Generate the dataset
dataset = generate_samples(0.5, 1.5, 100)

X = dataset.iloc[:, :2].values       # [X1kₖ, X2kₖ]
y = dataset.iloc[:, 2:].values      # [X1k+1, X2k+1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x1_tree = RegressionTree(X_train, y_train[:, 0], max_depth=2, control_by="depth")
x2_tree = RegressionTree(X_train, y_train[:, 1], max_depth=2, control_by="depth")

# Predict on test set
x1_prediction = np.array(x1_tree.predict(X_test)).reshape(-1, 1)
x2_prediction = np.array(x2_tree.predict(X_test)).reshape(-1, 1)
predictions = np.hstack((x1_prediction, x2_prediction))

print("Mean squared error: ", mean_squared_error(y_test, predictions))

# Plot test predictions
predictions_df = pd.DataFrame(predictions, columns=["x1_future_state", "x2_future_state"])

plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 0], label="Actual x1ₖ₊₁")
plt.plot(predictions_df["x1_future_state"], label="Predicted x1ₖ₊₁",linestyle="--")
plt.xlabel("Sample")
plt.ylabel("x1")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 1], label="Actual x2ₖ₊₁")
plt.plot(predictions_df["x2_future_state"], label="Predicted x2ₖ₊₁",linestyle="--")
plt.xlabel("Sample")
plt.ylabel("x2")
plt.legend()
plt.show()

#