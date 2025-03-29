import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from RegressionTree import RegressionTree

#Task 4.1
def generate_samples(n_samples=1000):
    np.random.seed(42)
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []

    for _ in range(n_samples):
        x1 = np.random.uniform(-5, 5)
        x2 = np.random.uniform(-5, 5)
        # using the formula from task3
        x1_next_state = 0.9 * x1 - 0.2 * x2
        x2_next_state = 0.2 * x1 + 0.9 * x2

        x1_list.append(x1)
        x2_list.append(x2)
        y1_list.append(x1_next_state)
        y2_list.append(x2_next_state)

    data = np.hstack((
        np.array(x1_list).reshape(-1, 1),
        np.array(x2_list).reshape(-1, 1),
        np.array(y1_list).reshape(-1, 1),
        np.array(y2_list).reshape(-1, 1)
    ))
    df = pd.DataFrame(data)
    #df.to_csv("q4_part1_dataset.csv", index=False, header=False)
    return df

# Generate the dataset
dataset = generate_samples(2000)

X = dataset.iloc[:, :2].values       # [X1kₖ, X2kₖ]
y = dataset.iloc[:, 2:].values      # [X1k+1, X2k+1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x1_tree = RegressionTree(X_train, y_train[:, 0], max_depth=10, control_by="depth")
x2_tree = RegressionTree(X_train, y_train[:, 1], max_depth=10, control_by="depth")

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
plt.xlabel("Steps")
plt.ylabel("x1")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 1], label="Actual x2ₖ₊₁")
plt.plot(predictions_df["x2_future_state"], label="Predicted x2ₖ₊₁",linestyle="--")
plt.xlabel("Steps")
plt.ylabel("x2")
plt.legend()
plt.show()


x_current = np.random.uniform(-5,5,size=2)
print(f"Initial state {x_current}")

# Store trajectories
predicted_trajectory = [x_current.copy()]
true_trajectory = [x_current.copy()]

for _ in range(20):
    # Predict next state using the trained trees
    x1_next = x1_tree.predict([x_current])[0]
    x2_next = x2_tree.predict([x_current])[0]
    x_next_pred = np.array([x1_next, x2_next])

    # Ground truth next state using the known system model
    x1_true = 0.9 * x_current[0] - 0.2 * x_current[1]
    x2_true = 0.2 * x_current[0] + 0.9 * x_current[1]
    x_next_true = np.array([x1_true, x2_true])

    # Append to trajectories
    predicted_trajectory.append(x_next_pred.copy())
    true_trajectory.append(x_next_true.copy())
    #Only for pred
    x_current = x_next_pred

predicted_trajectory = np.array(predicted_trajectory)
true_trajectory = np.array(true_trajectory)

plt.figure(figsize=(10, 5))
plt.plot(true_trajectory[:, 0], label="True x1", linewidth=2)
plt.plot(predicted_trajectory[:, 0], label="Predicted x1", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("x1")
plt.title("x1 Trajectory")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(true_trajectory[:, 1], label="True x2", linewidth=2)
plt.plot(predicted_trajectory[:, 1], label="Predicted x2", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("x2")
plt.title("x2 Trajectory")
plt.legend()
plt.grid(True)
plt.show()
