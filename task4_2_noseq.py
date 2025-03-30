import numpy as np
from RegressionTree import RegressionTree
import matplotlib.pyplot as plt

def generate_program_data(n_samples=2000):
    np.random.seed(42)
    X = []
    y = []

    for _ in range(n_samples):
        x = np.random.uniform(-3, 3)
        z = np.random.uniform(0, 15)

        if x > 1:
            x_next = 0
        else:
            x_next = x + 0.2
        z_next = z + x_next

        X.append([x, z])
        y.append([x_next, z_next])

    return np.array(X), np.array(y)

# Train model
X_train, y_train = generate_program_data(2500)

x_tree = RegressionTree(X_train, y_train[:, 0], max_depth=13, control_by="depth")
z_tree = RegressionTree(X_train, y_train[:, 1], max_depth=13, control_by="depth")

x = np.random.uniform(-3, 3)
z = np.random.uniform(0, 15)
initial_true_x = x
initial_true_z = z

pred_traj = [(x, z)]
true_traj = [(x, z)]

for _ in range(20):
    x_pred = x_tree.predict([[x, z]])[0]
    z_pred = z_tree.predict([[x, z]])[0]
    pred_traj.append((x_pred, z_pred))

    x, z = x_pred, z_pred

x = initial_true_x
z = initial_true_z
for _ in range(20):
    x_true = 0 if x > 1 else x + 0.2
    z_true = z + x_true
    true_traj.append((x_true, z_true))

    x, z = x_true, z_true

pred_traj = np.array(pred_traj)
true_traj = np.array(true_traj)

plt.figure(figsize=(10, 5))
plt.plot(true_traj[:, 0], label="True x")
plt.plot(pred_traj[:, 0], label="Pred x", linestyle="--")
plt.title("Task 4.2 - x Trajectory")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(true_traj[:, 1], label="True z")
plt.plot(pred_traj[:, 1], label="Pred z", linestyle="--")
plt.title("Task 4.2 - z Trajectory")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(true_traj, pred_traj)
print(f"MSE (scikit-learn): {mse:.6f}")

manual_mse = np.mean((true_traj - pred_traj) ** 2)
print(f"MSE (manual): {manual_mse:.6f}")
