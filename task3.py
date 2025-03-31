import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

from RegressionTree import RegressionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#generate the training dataset
#store inputs [xk,vk][xk+1,vk+1]
def generate_samples(n_samples=1000):
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []

    for _ in range(n_samples):
        x1 = np.random.uniform(-10, 10)
        x2 = 10  # velocity is constant
        # using the formula from task3
        x1_next = x1 + 0.1 * x2
        x2_next = 10

        x1_list.append(x1)
        x2_list.append(x2)
        y1_list.append(x1_next)
        y2_list.append(x2_next)

    data = np.hstack((
        np.array(x1_list).reshape(-1, 1),
        np.array(x2_list).reshape(-1, 1),
        np.array(y1_list).reshape(-1, 1),
        np.array(y2_list).reshape(-1, 1)
    ))
    df = pd.DataFrame(data)
    df.to_csv("task3_dataset.csv", index=False, header=False)
    return df

#split into training and testing datasets 80% 20%
task3_data = generate_samples(1000)
X = task3_data.iloc[:,:2].values
y = task3_data.iloc[:,2:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train two regression trees
# RT1 to train Xk+1
x_tree = RegressionTree(X_train, y_train[:,0])
#RT2 to train vk+1
v_tree= RegressionTree(X_train, y_train[:,1])
#Predict on test set

x_prediction = np.array(x_tree.predict(X_test)).reshape(-1, 1)
v_prediction = np.array(v_tree.predict(X_test)).reshape(-1, 1)
predictions = np.hstack((x_prediction, v_prediction))

print("Mean squared error: ", mean_squared_error(y_test, predictions))

predictions_df = pd.DataFrame(predictions, columns=["x_future_state", "v_future_state"])

plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 0], label="Actual x1_k+1")
plt.plot(predictions_df["x_future_state"], label="Predicted x1_k+1", linestyle ='--' )
plt.xlabel("Sample")
plt.ylabel("x")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 1], label="Actual V_k+1")
plt.plot(predictions_df["v_future_state"], label="Predicted V_k+1",linestyle ='--')
plt.xlabel("Sample")
plt.ylabel("v")
plt.legend()
plt.show()
