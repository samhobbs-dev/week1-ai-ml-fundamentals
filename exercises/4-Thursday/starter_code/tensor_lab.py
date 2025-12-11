"""
Exercise 07: Tensor Manipulation Lab
====================================

Practice TensorFlow tensor operations.
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("TENSOR MANIPULATION LAB")
print(f"TensorFlow version: {tf.__version__}")
print("=" * 60)

# =============================================================================
# PART 1: TENSOR CREATION
# =============================================================================

print("\n--- PART 1: TENSOR CREATION ---")

# TODO 1: A scalar (rank 0) with value 42
scalar = 42
# print(f"Scalar: {scalar}, shape: {scalar.shape}")

# TODO 2: A vector (rank 1) with values [1, 2, 3, 4, 5]
vector = [1, 2, 3, 4, 5]
# print(f"Vector: {vector}, shape: {vector.shape}")

# TODO 3: A 3x3 matrix (rank 2) of all ones
matrix_ones = tf.ones([3, 3])
# print(f"Matrix ones shape: {matrix_ones.shape}")

# TODO 4: A 2x3x4 tensor (rank 3) of zeros
tensor_3d = tf.zeros([2, 3, 4])
# print(f"3D tensor shape: {tensor_3d.shape}")

# TODO 5: A tensor from a NumPy array
np_array = np.array([[1, 2], [3, 4], [5, 6]])
from_numpy = tf.convert_to_tensor(np_array)
# print(f"From NumPy shape: {from_numpy.shape}")

# TODO 6: A random normal tensor with shape (100, 10)
random_normal = tf.random.normal([100, 10], mean=0, stddev=1)
# print(f"Random normal shape: {random_normal.shape}")

# TODO 7: A tensor with values from 0 to 99
range_tensor = tf.range(100)
# print(f"Range tensor: {range_tensor}")

# =============================================================================
# PART 2: SHAPE MANIPULATION
# =============================================================================

print("\n--- PART 2: SHAPE MANIPULATION ---")

original = tf.range(24)
print(f"Original: {original.shape}")

# TODO: Reshape to (4, 6)
shape_a = tf.reshape(original, [4, 6])
# print(f"Reshaped to (4, 6): {shape_a.shape}")

# TODO: Reshape to (2, 3, 4)
shape_b = tf.reshape(original, [2, 3, 4])

# TODO: Reshape to (24, 1)
shape_c = tf.reshape(original, [24, 1])

# TODO: Reshape using -1
reshape_auto = tf.reshape(original, [6, -1])
# print(f"Shape with -1 (6, -1): {reshape_auto.shape}")

# Expand dims
vector_test = tf.constant([1, 2, 3, 4])
# TODO: Add dimension at axis 0
expanded_0 = tf.expand_dims(vector_test, axis=0)
# print(f"Expanded at axis 0: {expanded_0.shape}")

# TODO: Add dimension at axis 1
expanded_1 = tf.expand_dims(vector_test, axis=1)
# print(f"Expanded at axis 1: {expanded_1.shape}")

# =============================================================================
# PART 3: BROADCASTING
# =============================================================================

print("\n--- PART 3: BROADCASTING ---")

a = tf.constant([[1, 2, 3],
                 [4, 5, 6]])  # Shape: (2, 3)

b = tf.constant([10, 20, 30])  # Shape: (3,)

# TODO: Add a and b
result = tf.add(a, b)
# print(f"a shape: {a.shape}")
# print(f"b shape: {b.shape}")
# print(f"Result shape: {result.shape}")
# print(f"Result:\n{result}")

# Predict result shapes before running!
# Scenario 1
a1 = tf.ones([3, 1])
b1 = tf.ones([1, 4])
# Predicted shape: 2,2 (actually 3,4)
c1 = tf.add(a1, b1)
print(f"Scenario 1 result shape: {c1.shape}")

# Scenario 2
a2 = tf.ones([5, 3, 1])
b2 = tf.ones([1, 4])
# Predicted shape: 5,3,4
c2 = a2 + b2
print(f"Scenario 2 result shape: {c2.shape}")

# =============================================================================
# PART 4: COMMON OPERATIONS
# =============================================================================

print("\n--- PART 4: COMMON OPERATIONS ---")

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

# TODO: Math operations
# mean_all = tf.reduce_mean(x)
# mean_rows = tf.reduce_mean(x, axis=1)
# mean_cols = tf.reduce_mean(x, axis=0)
# sum_all = tf.reduce_sum(x)
# max_val = tf.reduce_max(x)
# argmax_rows = tf.argmax(x, axis=1)

# Matrix operations
A = tf.constant([[1., 2.],
                 [3., 4.]])

B = tf.constant([[5., 6.],
                 [7., 8.]])

# TODO: Matrix multiplication
matmul_result = tf.matmul(A, B)  # or A @ B

# TODO: Element-wise multiplication
elementwise = A * B

# TODO: Transpose
transposed = tf.transpose(A)

# =============================================================================
# CHALLENGE PROBLEMS
# =============================================================================

print("\n--- CHALLENGE PROBLEMS ---")

# Challenge A: Batch Normalization Prep
# images = tf.random.normal([2, 4, 4, 3])
# TODO: Normalize each channel

# Challenge B: Manual Softmax
def manual_softmax(logits):
    """Implement softmax without tf.nn.softmax"""
    # TODO: Implement
    pass

# Challenge C: Fix Shape Errors
# TODO: Fix these operations

