# Generate training data

import numpy as np
from numba.cuda.random import xoroshiro128p_next


def generate_vehicle_data(num_samples=1000):
    x_vals = np.random.uniform(-5,5,size=num_samples)
    v_vals = np.full_like(x_vals,10)
    X = np.column_stack((x_vals, v_vals))

    x_next = x_vals + 0.1 * v_vals
    v_next = np.full_like(x_next,10.0)
    y = np.column_stack((x_next, v_next))

    return X, y

print(generate_vehicle_data(5))