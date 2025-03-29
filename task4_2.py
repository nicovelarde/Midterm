import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RegressionTree import RegressionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Task 4.2
#generate training data from it
def generate_samples(n_samples=1000):
    np.random.seed(42)

    X = []
    y = []

    for _ in range(n_samples):
        x = np.random.uniform(-3,3)
        z = np.random.uniform(0,15)

        #use function from problem
        if x > 1:
            x_next = 0
        else:
            x_next = x + 0.2
        z_next = z + x_next

        # keep current and next states
        X.append([x,z])
        y.append([x_next, z_next])

    return np.array(X), np.array(y)

X,y = generate_samples(2000)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Train two regression trees
x_tree = RegressionTree(X_train, y_train[:,0], max_depth=10, control_by='depth')
z_tree = RegressionTree(X_train, y_train[:,1], max_depth=10, control_by='depth')

#Predict the future trajectory of x=2 and x=0 for 20 steps
#initial state

x = np.random.uniform(-3,3)
z = np.random.uniform(0,15)

predicted_trajectory = [(x,z)]
true_trajectory = [(x,z)]

for _ in range(20):
    x_pred = x_tree.predict([[x, z]])[0]
    z_pred = z_tree.predict([[x, z]])[0]
    predicted_trajectory.append((x_pred, z_pred))

    # true next state
    if x > 1:
        x_true = 0
    else:
        x_true = x + 0.2
    z_true = z + x_true
    true_trajectory.append((x_true, z_true))

    x,z = x_pred, z_pred

predicted_trajectory = np.array(predicted_trajectory)
true_trajectory = np.array(true_trajectory)

# Plot x
plt.figure(figsize=(10, 5))
plt.plot(true_trajectory[:, 0], label="True x", linewidth=2)
plt.plot(predicted_trajectory[:, 0], label="Predicted x", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("x")
plt.title("x Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# Plot z
plt.figure(figsize=(10, 5))
plt.plot(true_trajectory[:, 1], label="True z", linewidth=2)
plt.plot(predicted_trajectory[:, 1], label="Predicted z", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("z")
plt.title("z Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# Optional: Print final error
mse = mean_squared_error(true_trajectory, predicted_trajectory)
print(f"MSE over 20-step prediction: {mse:.6f}")

