from RegressionTree import RegressionTree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

def generate_data(samples, limit_dwn, limit_up, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(limit_dwn, limit_up,samples).reshape(-1,1)
    y = 0.8 * np.sin(X - 1)
    return X , y

X,y = generate_data(100,-3,3,42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, train_size=0.8, random_state=42)

"""
test_result
    - time cost of building the tree (time python) 
    - Normal test error (MSE)
    - Scikit learn test error (MSE)
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

y_pred = tree.predict(X_test)
normal_mse = np.mean((y_test - y_pred)**2)
scikit_learn_mse = mean_squared_error(y_test, y_pred)

height = tree.height()

print(f"\n No Limitation:")
print(f"  - Tree Height: {height}")
print(f"  - Scikit learn test error: {scikit_learn_mse:.5f}")
print(f"  - Normal mse test error: {normal_mse:.5f}")
print(f"  - Build Time: {total_time:.5f} sec")
# Task 2.2
print("================================================Task 2.2 ==================================")
for fraction in [0.5, 0.75]:
    max_height = int(height*fraction)

    start_time = time.time()
    tree=RegressionTree(X_train,y_train,max_depth=max_height, control_by='depth')
    stop_time = time.time()
    total_time = stop_time-start_time

    y_pred = tree.predict(X_test)
    normal_mse = np.mean((y_test - y_pred) ** 2)
    scikit_learn_mse = mean_squared_error(y_test, y_pred)

    curr_height = tree.height()

    print(f"\n Limited Height ({fraction * 100:.0f}% of full height):")
    print(f"  - Max Height: {curr_height}")
    print(f"  - Scikit learn test error: {scikit_learn_mse:.5f}")
    print(f"  - Normal mse test error: {normal_mse:.5f}")
    print(f"  - Build Time: {total_time:.5f} sec")


#Task 2.3
print("================================================Task 2.3 ==================================")
for leaf_size in [2, 4, 8]:
    start_time = time.time()
    tree = RegressionTree(X_train, y_train, leaf_size, control_by='leaf')
    stop_time = time.time()
    total_time = stop_time - start_time

    y_pred = tree.predict(X_test)
    normal_mse = np.mean((y_test - y_pred) ** 2)
    scikit_learn_mse = mean_squared_error(y_test, y_pred)

    curr_height = tree.height()

    print(f"\n Limited Leaf Size ({leaf_size} per leaf):")
    print(f"  - Max Height: {curr_height}")
    print(f"  - Scikit learn test error: {scikit_learn_mse:.5f}")
    print(f"  - Normal mse test error: {normal_mse:.5f}")
    print(f"  - Build Time: {total_time:.5f} sec")

