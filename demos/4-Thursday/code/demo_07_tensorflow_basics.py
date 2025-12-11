"""
Demo 07: TensorFlow Basics
==========================
Week 1, Thursday - TensorFlow and Keras

Introduction to TensorFlow: tensors, operations, shapes, and GPU detection.
This bridges yesterday's NumPy implementations to TensorFlow.

INSTRUCTOR NOTES:
- Show the ecosystem diagram first
- Emphasize that TensorFlow is like "NumPy on steroids"
- GPU detection is exciting for trainees with NVIDIA cards

Estimated Time: 15-20 minutes
"""

import tensorflow as tf
import numpy as np

# =============================================================================
# SECTION 1: TENSORFLOW INTRODUCTION
# =============================================================================
print("=" * 60)
print("DEMO 07: TENSORFLOW BASICS")
print("The Foundation for Deep Learning")
print("=" * 60)

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Eager Execution: {tf.executing_eagerly()}")

# =============================================================================
# SECTION 2: GPU DETECTION
# =============================================================================
print("\n--- GPU DETECTION ---")

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"CPUs available: {len(cpus)}")
print(f"GPUs available: {len(gpus)}")

if gpus:
    print("\nGPU Details:")
    for gpu in gpus:
        print(f"  {gpu}")
    print("\nTensorFlow will automatically use the GPU for computations!")
else:
    print("\nNo GPU detected. Running on CPU.")
    print("(This is fine for learning - production training benefits from GPU)")

# =============================================================================
# SECTION 3: CREATING TENSORS
# =============================================================================
print("\n--- CREATING TENSORS ---")

# From Python lists
tensor_list = tf.constant([1, 2, 3, 4, 5])
print(f"From list: {tensor_list}")
print(f"  Shape: {tensor_list.shape}, Dtype: {tensor_list.dtype}")

# From NumPy (seamless integration!)
numpy_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
tensor_numpy = tf.constant(numpy_array)
print(f"\nFrom NumPy:\n{tensor_numpy}")
print(f"  Shape: {tensor_numpy.shape}, Dtype: {tensor_numpy.dtype}")

# Special tensors
zeros = tf.zeros([3, 4])
ones = tf.ones([2, 3])
random_normal = tf.random.normal([3, 3], mean=0, stddev=1)
random_uniform = tf.random.uniform([2, 4], minval=0, maxval=10)

print(f"\nZeros (3x4):\n{zeros}")
print(f"\nOnes (2x3):\n{ones}")
print(f"\nRandom Normal (3x3):\n{random_normal}")
print(f"\nRandom Uniform (2x4):\n{random_uniform}")

# =============================================================================
# SECTION 4: TENSOR PROPERTIES
# =============================================================================
print("\n--- TENSOR PROPERTIES ---")

t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

print(f"Tensor:\n{t}")
print(f"Shape: {t.shape}")
print(f"Dtype: {t.dtype}")
print(f"Rank (ndim): {len(t.shape)}")
print(f"Number of elements: {tf.size(t).numpy()}")
print(f"Device: {t.device}")

# Convert to NumPy
numpy_version = t.numpy()
print(f"\nAs NumPy array:\n{numpy_version}")
print(f"Type: {type(numpy_version)}")

# =============================================================================
# SECTION 5: BASIC OPERATIONS
# =============================================================================
print("\n--- BASIC OPERATIONS ---")

a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

print(f"a = {a.numpy()}")
print(f"b = {b.numpy()}")

# Element-wise operations
print(f"\na + b = {(a + b).numpy()}")
print(f"a - b = {(a - b).numpy()}")
print(f"a * b = {(a * b).numpy()}")  # Element-wise
print(f"a / b = {(a / b).numpy()}")
print(f"a ** 2 = {(a ** 2).numpy()}")

# Reduction operations
print(f"\ntf.reduce_sum(a) = {tf.reduce_sum(a).numpy()}")
print(f"tf.reduce_mean(a) = {tf.reduce_mean(a).numpy()}")
print(f"tf.reduce_max(a) = {tf.reduce_max(a).numpy()}")

# =============================================================================
# SECTION 6: MATRIX OPERATIONS
# =============================================================================
print("\n--- MATRIX OPERATIONS ---")

A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

print(f"Matrix A:\n{A.numpy()}")
print(f"Matrix B:\n{B.numpy()}")

# Matrix multiplication (three ways)
matmul1 = tf.matmul(A, B)
matmul2 = A @ B  # Python 3.5+ operator
print(f"\nA @ B (matrix multiply):\n{matmul1.numpy()}")

# Transpose
transposed = tf.transpose(A)
print(f"\ntf.transpose(A):\n{transposed.numpy()}")

# Element-wise multiply (not matmul!)
elementwise = A * B
print(f"\nA * B (element-wise):\n{elementwise.numpy()}")

# =============================================================================
# SECTION 7: SHAPE MANIPULATION
# =============================================================================
print("\n--- SHAPE MANIPULATION ---")

# Original tensor
original = tf.range(12)
print(f"Original: {original.numpy()}")
print(f"Shape: {original.shape}")

# Reshape
reshaped_2d = tf.reshape(original, [3, 4])
reshaped_3d = tf.reshape(original, [2, 2, 3])

print(f"\nReshaped to (3, 4):\n{reshaped_2d.numpy()}")
print(f"\nReshaped to (2, 2, 3):\n{reshaped_3d.numpy()}")

# Using -1 for automatic dimension
auto_reshape = tf.reshape(original, [4, -1])  # -1 infers 3
print(f"\nReshape with -1 (4, -1):\n{auto_reshape.numpy()}")
print(f"Shape: {auto_reshape.shape}")

# Expand and squeeze dimensions
vector = tf.constant([1, 2, 3])
print(f"\nOriginal vector: {vector.numpy()}, shape: {vector.shape}")

expanded = tf.expand_dims(vector, axis=0)
print(f"Expanded at axis 0: {expanded.numpy()}, shape: {expanded.shape}")

expanded = tf.expand_dims(vector, axis=1)
print(f"Expanded at axis 1:\n{expanded.numpy()}\nshape: {expanded.shape}")

# =============================================================================
# SECTION 8: VARIABLES (TRAINABLE TENSORS)
# =============================================================================
print("\n--- TENSORFLOW VARIABLES ---")
print("Variables are mutable tensors - used for weights!")

# Create a variable
weights = tf.Variable(tf.random.normal([3, 2]), name='weights')
print(f"Initial weights:\n{weights.numpy()}")

# Update variable
weights.assign(tf.zeros([3, 2]))
print(f"\nAfter assign zeros:\n{weights.numpy()}")

weights.assign_add(tf.ones([3, 2]))
print(f"\nAfter assign_add ones:\n{weights.numpy()}")

# Variables track gradients (preview of training)
print(f"\ntrainable: {weights.trainable}")
print(f"name: {weights.name}")

# =============================================================================
# SECTION 9: AUTOMATIC DIFFERENTIATION
# =============================================================================
print("\n--- AUTOMATIC DIFFERENTIATION (GradientTape) ---")
print("This is how TensorFlow computes gradients for backpropagation!")

x = tf.Variable(3.0)

# Record operations for gradient computation
with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1

# Compute gradient: dy/dx
gradient = tape.gradient(y, x)

print(f"y = x^2 + 2x + 1")
print(f"At x = {x.numpy()}:")
print(f"  y = {y.numpy()}")
print(f"  dy/dx = 2x + 2 = {gradient.numpy()} (expected: 8)")

# Multiple variables
print("\n--- Multiple Variables ---")
w = tf.Variable(2.0)
b = tf.Variable(1.0)
x_input = tf.constant(3.0)

with tf.GradientTape() as tape:
    y_pred = w * x_input + b  # Linear: y = wx + b

gradients = tape.gradient(y_pred, [w, b])
print(f"y = w*x + b where w={w.numpy()}, b={b.numpy()}, x={x_input.numpy()}")
print(f"dy/dw = x = {gradients[0].numpy()}")
print(f"dy/db = 1 = {gradients[1].numpy()}")

# =============================================================================
# SECTION 10: COMPARISON: TENSORFLOW VS NUMPY
# =============================================================================
print("\n--- TENSORFLOW vs NUMPY ---")

import time

size = 1000

# Create data
np_a = np.random.randn(size, size).astype(np.float32)
np_b = np.random.randn(size, size).astype(np.float32)
tf_a = tf.constant(np_a)
tf_b = tf.constant(np_b)

# NumPy timing
start = time.time()
for _ in range(10):
    np_result = np.matmul(np_a, np_b)
numpy_time = time.time() - start

# TensorFlow timing
start = time.time()
for _ in range(10):
    tf_result = tf.matmul(tf_a, tf_b)
tf_time = time.time() - start

print(f"Matrix multiplication ({size}x{size}) - 10 iterations:")
print(f"  NumPy time: {numpy_time:.4f}s")
print(f"  TensorFlow time: {tf_time:.4f}s")

if gpus:
    print(f"  (TensorFlow is using GPU!)")

# =============================================================================
# SECTION 11: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. TENSORS are multi-dimensional arrays with uniform dtype
   - tf.constant() for immutable data
   - tf.Variable() for trainable parameters (weights)

2. SHAPES matter!
   - (batch, features) for Dense layers
   - (batch, height, width, channels) for CNNs
   - Use tf.reshape(), tf.expand_dims(), tf.squeeze()

3. OPERATIONS work element-wise by default
   - Use tf.matmul() or @ for matrix multiplication
   - Broadcasting works like NumPy

4. GradientTape enables automatic differentiation
   - Records operations during forward pass
   - Computes gradients for backpropagation

5. GPU ACCELERATION is automatic
   - TensorFlow uses available GPU without code changes
   - Same code runs on CPU, GPU, or TPU

NEXT: Keras makes building models even easier!
""")

