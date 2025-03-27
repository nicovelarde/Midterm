from RegressionTree import RegressionTree
import numpy as np
from sklearn.model_selection import train_test_split
import time

def generate_data(samples, limit_up, limit_dwn, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(limit_dwn, limit_up,samples).reshape(-1,1)
    y = 0.8 * np.sin(x - 1)
    return x , y

X,y = generate_data(100,3,-3,42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, train_size=0.8, random_state=42)

"""
test_result
    - time cost of building the tree (time python) 
    - test error (MSE)
    - height of the regression tree
"""

#Tasks 2:

# Task 2.1
# No limitation
print("================================================Task 2.1 ==================================")

start_time = time.time()
tree = RegressionTree(X_train, y_train)
stop_time = time.time()
total_time = stop_time-start_time
print(f"The time cost of building the tree is {total_time:.4f}")

y_pred = tree.predict(X_test)
mse = np.mean((y_test - y_pred)**2)
print(f"The test error is {mse:.4f}")

height = tree.height()
print(f"The height of the resulting regression tree is {height}")

# Task 2.2
print("================================================Task 2.2 ==================================\n")
print("======================Using 1/2 height=====================================\n")
start_time = time.time()
tree2=RegressionTree(X_train,y_train,max_depth=7, control_by='depth')
stop_time = time.time()
total_time = stop_time-start_time
print(f"The time cost of building the tree with 1/2 height is {total_time:.4f}")

y_pred = tree2.predict(X_test)
mse = np.mean((y_test - y_pred)**2)
print(f"The test error is {mse:.4f}")

height = tree2.height()
print(f"The height of the resulting regression tree is {height}")

print("======================Using 3/4 height=====================================\n")
start_time = time.time()
tree3=RegressionTree(X_train,y_train,max_depth=(14*(3/4)), control_by='depth')
stop_time = time.time()
total_time = stop_time - start_time
print(f"The time cost of building the tree with 3/4 height is {total_time:.4f}")

y_pred = tree3.predict(X_test)
mse = np.mean((y_test - y_pred)**2)
print(f"The test error is {mse:.4f}")

height2 = tree3.height()
print(f"The height of the resulting regression tree is {height2}")

#Task 2.3
print("================================================Task 2.3 ==================================\n")
print("======================Using a leaf size limit of 2 =====================================\n")
start_time = time.time()
tree4 = RegressionTree(X_train,y_train,min_samples_leaf=2,control_by='leaf')
stop_time=time.time()
total_time=stop_time - start_time
print(f"The time cost of building the tree with a limit of 2 in leaf size is {total_time:.4f}")

y_pred = tree4.predict(X_test)
mse = np.mean((y_test - y_pred)**2)
print(f"The test error is {mse:.4f}")

height = tree4.height()
print(f"The height of the resulting tree is {height}")

print("======================Using a leaf size limit of 4 =====================================\n")
start_time = time.time()
tree5 = RegressionTree(X_train,y_train,min_samples_leaf=4,control_by='leaf')
stop_time=time.time()
total_time=stop_time - start_time
print(f"The time cost of building the tree with a limit of 4 in leaf size is {total_time:.4f}")

y_pred = tree5.predict(X_test)
mse = np.mean((y_test - y_pred)**2)
print(f"The test error is {mse:.4f}")

height = tree5.height()
print(f"The height of the resulting tree is {height}")

print("======================Using a leaf size limit of 8 =====================================\n")
start_time = time.time()
tree6 = RegressionTree(X_train,y_train,min_samples_leaf=8,control_by='leaf')
stop_time=time.time()
total_time=stop_time - start_time
print(f"The time cost of building the tree with a limit of 8 in leaf size is {total_time:.4f}")

y_pred = tree6.predict(X_test)
mse = np.mean((y_test - y_pred)**2)
print(f"The test error is {mse:.4f}")

height = tree6.height()
print(f"The height of the resulting tree is {height}")