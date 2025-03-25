from RegressionTree import RegressionTree
import numpy as np
from sklearn.model_selection import train_test_split
import time

# test using y=0.8sin(x-1)
# x e [-1,3]
# 100 training samples - uniformly distributed
# make an 80% 20% split for training-testing

def generate_data(samples, limit_up, limit_dwn, seed=42):
    np.random.seed(seed)
    #Generate uniform samples:
    x = np.random.uniform(limit_dwn, limit_up,samples).reshape(-1,1)
    #compute y
    y = 0.8 * np.sin(x - 1)

    return x , y

X,y = generate_data(100,3,-3,42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, train_size=0.8, random_state=42)

# print(f"X_train: {X_train.shape}")
# print(f"y_train: {y_train.shape}")

"""
test_result
    - time cost of building the tree (time python) 
    - test error (MSE)
    - height of the regression tree
"""

# Task 2.1
# No limitation
#time cost
start_time = time.time()
tree= RegressionTree(X_train, y_train)
stop_time = time.time()
total_time = stop_time - start_time
print(f"Total time for 2.1 is {total_time:.4f}")

y_pred = tree.predict(X_test)
mse= np.mean((y_test - y_pred) **2)
print(f"The test error for 2.1 is {mse:.4f}")

height = tree.height()
print(f"The height for 2.1 is {height}")

# Task 2.2
max_depth_half = int(height / 2)
max_depth_three_quarters = int(height * 0.75)
print("Half Depth:", max_depth_half)
print("3/4 Depth:", max_depth_three_quarters)

start_time = time.time()
tree= RegressionTree(X_train, y_train, max_depth_half, control_by="depth" )
stop_time = time.time()
total_time = stop_time - start_time
print(f"Total time for 2.2 with 1/2 depth is {total_time:.4f}")

y_pred = tree.predict(X_test)
mse= np.mean((y_test - y_pred) **2)
print(f"The test error for 2.2 with 1/2 depth is {mse:.4f}")

height = tree.height()
print(f"The height for 2.2 with half depth is {height}")

start_time = time.time()
tree= RegressionTree(X_train, y_train, max_depth_three_quarters, control_by="depth")
stop_time = time.time()
total_time = stop_time - start_time
print(f"Total time for 2.2 with 3/4 depth is {total_time:.4f}")

y_pred = tree.predict(X_test)
mse= np.mean((y_test - y_pred) **2)
print(f"The test error for 2.2 with 3/4 depth is {mse:.4f}")

height = tree.height()
print(f"The height for 2.2 with 3/4 depth is {height}")
# Task 2.3






