import numpy as np
import matplotlib.pyplot as plt
from RegressionTree import RegressionTree

def generate_linear_data(n_samples=2000):
    np.random.seed(42)
    X = []
    y = []

    for _ in range(n_samples):
        x1 = np.random.uniform(-5, 5)
        x2 = np.random.uniform(-5, 5)
        x1_next = 0.9 * x1 - 0.2 * x2
        x2_next = 0.2 * x1 + 0.9 * x2

        X.append([x1, x2])
        y.append([x1_next, x2_next])

    return np.array(X), np.array(y)

X_train, y_train = generate_linear_data()

x1_tree = RegressionTree(X_train, y_train[:, 0], max_depth=10, control_by="depth")
x2_tree = RegressionTree(X_train, y_train[:, 1], max_depth=10, control_by="depth")

# Recursive simulation
x = np.random.uniform(-5, 5,2)  # initial state
initial_x_true = x.copy()
predicted_traj = [x.copy()]
true_traj = [x.copy()]

for _ in range(20):
    x_pred_1 = x1_tree.predict([x])[0]
    x_pred_2 = x2_tree.predict([x])[0]
    x_pred = np.array([x_pred_1, x_pred_2])
    predicted_traj.append(x_pred)
    x = x_pred  # use predicted for next step

x = initial_x_true
for _ in range(0,20):
    # ground truth using known equation
    x_true = np.array([
        0.9 * x[0] - 0.2 * x[1],
        0.2 * x[0] + 0.9 * x[1]
    ])
    true_traj.append(x_true)
    x = x_true  # use predicted for next step

predicted_traj = np.array(predicted_traj)
true_traj = np.array(true_traj)

plt.figure(figsize=(10, 5))
plt.plot(true_traj[:, 0], label="True x1")
plt.plot(predicted_traj[:, 0], label="Pred x1", linestyle="--")
plt.title("Task 4.1 - x1 Trajectory")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(true_traj[:, 1], label="True x2")
plt.plot(predicted_traj[:, 1], label="Pred x2", linestyle="--")
plt.title("Task 4.1 - x2 Trajectory")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(true_traj, predicted_traj)
print(f"MSE (scikit-learn): {mse:.6f}")

manual_mse = np.mean((true_traj - predicted_traj) ** 2)
print(f"MSE (manual): {manual_mse:.6f}")