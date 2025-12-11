"""
Demo 09: Inside Dense Layers
============================
Week 1, Thursday - TensorFlow and Keras

Deep dive into Dense layers: extracting weights, 
manual forward pass, and verifying Keras matches our math.

INSTRUCTOR NOTES:
- Connect to Wednesday's forward propagation
- Show that Keras does exactly what we did manually
- This builds confidence that they understand what's happening

Estimated Time: 15-20 minutes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# =============================================================================
# SECTION 1: INTRODUCTION
# =============================================================================
print("=" * 60)
print("DEMO 09: INSIDE DENSE LAYERS")
print("Verifying Keras Does What We Learned")
print("=" * 60)

print("""
Yesterday we computed forward propagation manually:
    z = W @ x + b
    a = activation(z)

Today we'll prove that Keras Dense layers do the same thing!
""")

# =============================================================================
# SECTION 2: CREATE A SIMPLE LAYER
# =============================================================================
print("\n--- CREATING A DENSE LAYER ---")

# Create a Dense layer: 4 inputs -> 3 outputs
layer = layers.Dense(3, activation='relu', input_shape=(4,), name='my_dense')

# Build the layer (creates weights)
layer.build((None, 4))

print(f"Layer name: {layer.name}")
print(f"Input units: 4")
print(f"Output units: {layer.units}")
print(f"Number of parameters: {layer.count_params()}")

# =============================================================================
# SECTION 3: EXTRACT WEIGHTS
# =============================================================================
print("\n--- EXTRACTING WEIGHTS ---")

# Get weights
weights = layer.get_weights()
W = weights[0]  # Weight matrix
b = weights[1]  # Bias vector

print(f"Number of weight arrays: {len(weights)}")
print(f"\nWeight matrix (W) shape: {W.shape}")
print(f"W:\n{W}")
print(f"\nBias vector (b) shape: {b.shape}")
print(f"b: {b}")

# Parameter count verification
print(f"\nParameter count:")
print(f"  Weights: {W.size} ({W.shape[0]} x {W.shape[1]})")
print(f"  Biases: {b.size}")
print(f"  Total: {W.size + b.size} = {layer.count_params()}")

# =============================================================================
# SECTION 4: MANUAL VS KERAS FORWARD PASS
# =============================================================================
print("\n--- MANUAL VS KERAS FORWARD PASS ---")

# Create input
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
print(f"Input x: {x}")
print(f"Input shape: {x.shape}")

# Manual computation
z_manual = np.dot(x, W) + b
a_manual = np.maximum(0, z_manual)  # ReLU

print(f"\n--- Manual Calculation ---")
print(f"z = x @ W + b")
print(f"z = {z_manual}")
print(f"a = ReLU(z) = {a_manual}")

# Keras computation
a_keras = layer(x).numpy()

print(f"\n--- Keras Calculation ---")
print(f"layer(x) = {a_keras}")

# Verify they match
print(f"\n--- Verification ---")
print(f"Match: {np.allclose(a_manual, a_keras)}")

# =============================================================================
# SECTION 5: SETTING CUSTOM WEIGHTS
# =============================================================================
print("\n--- SETTING CUSTOM WEIGHTS ---")

# Create a layer with specific weights
custom_layer = layers.Dense(2, use_bias=True, input_shape=(3,))
custom_layer.build((None, 3))

# Define custom weights
custom_W = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
], dtype=np.float32)

custom_b = np.array([0.5, -0.5], dtype=np.float32)

# Set the weights
custom_layer.set_weights([custom_W, custom_b])

print("Custom weights set:")
print(f"W:\n{custom_W}")
print(f"b: {custom_b}")

# Test
test_input = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
print(f"\nTest input: {test_input}")

# Manual calculation
expected = np.dot(test_input, custom_W) + custom_b
print(f"\nManual calculation:")
print(f"  (1*1 + 2*0 + 3*1) + 0.5 = {1*1 + 2*0 + 3*1 + 0.5}")
print(f"  (1*0 + 2*1 + 3*1) - 0.5 = {1*0 + 2*1 + 3*1 - 0.5}")
print(f"  Expected: {expected}")

# Keras calculation
actual = custom_layer(test_input).numpy()
print(f"Keras output: {actual}")
print(f"Match: {np.allclose(expected, actual)}")

# =============================================================================
# SECTION 6: FULL MODEL WEIGHT INSPECTION
# =============================================================================
print("\n--- FULL MODEL WEIGHT INSPECTION ---")

# Create a small model
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,), name='hidden'),
    layers.Dense(3, activation='softmax', name='output')
])

# Print architecture and weights
print("Model Architecture:")
model.summary()

print("\n--- Layer-by-Layer Weights ---")
for layer in model.layers:
    print(f"\n{layer.name}:")
    weights = layer.get_weights()
    if weights:
        W, b = weights
        print(f"  Weight shape: {W.shape}")
        print(f"  Bias shape: {b.shape}")
        print(f"  Parameters: {W.size + b.size}")
        print(f"  Weight mean: {W.mean():.4f}, std: {W.std():.4f}")

# =============================================================================
# SECTION 7: TRACING A FORWARD PASS THROUGH THE MODEL
# =============================================================================
print("\n--- TRACING FORWARD PASS ---")

# Sample input
x = np.array([[0.5, 1.0, -0.5, 0.2]], dtype=np.float32)
print(f"Input: {x}")

# Get intermediate outputs
print("\nLayer by layer:")

# Access hidden layer output
hidden_layer = model.get_layer('hidden')
hidden_output = hidden_layer(x)
print(f"After hidden layer (ReLU):")
print(f"  Shape: {hidden_output.shape}")
print(f"  Values: {hidden_output.numpy()}")

# Final output
final_output = model(x)
print(f"\nFinal output (Softmax):")
print(f"  Shape: {final_output.shape}")
print(f"  Probabilities: {final_output.numpy()}")
print(f"  Sum: {final_output.numpy().sum():.6f} (should be 1.0)")

# =============================================================================
# SECTION 8: WEIGHT INITIALIZATION
# =============================================================================
print("\n--- WEIGHT INITIALIZATION ---")

print("""
Different initializations affect training!

Common initializers:
- 'glorot_uniform' (default): Good general choice
- 'he_normal': Recommended for ReLU
- 'zeros': All zeros (usually bad!)
- 'ones': All ones (usually bad!)
""")

# Compare initializations
initializers = ['glorot_uniform', 'he_normal', 'zeros']

for init_name in initializers:
    test_layer = layers.Dense(100, kernel_initializer=init_name)
    test_layer.build((None, 100))
    W = test_layer.get_weights()[0]
    print(f"\n{init_name}:")
    print(f"  Mean: {W.mean():.6f}")
    print(f"  Std: {W.std():.6f}")
    print(f"  Range: [{W.min():.4f}, {W.max():.4f}]")

# =============================================================================
# SECTION 9: CONNECTING TO MLP CONCEPTS
# =============================================================================
print("\n--- CONNECTING TO MLP CONCEPTS ---")

print("""
WEDNESDAY'S MLP MATH:       KERAS EQUIVALENT:
==========================================
z = W @ x + b              Dense layer (linear part)
a = activation(z)          Dense layer (activation part)

Input layer (no weights)   input_shape=(n,) in first Dense
Hidden layer              Dense(neurons, activation='relu')
Output layer              Dense(outputs, activation='sigmoid/softmax')

Weight matrix W           layer.kernel
Bias vector b             layer.bias
""")

# Access using Keras terminology
print("Keras weight attributes:")
layer = model.get_layer('hidden')
print(f"  kernel (W) shape: {layer.kernel.shape}")
print(f"  bias (b) shape: {layer.bias.shape}")
print(f"  trainable: {layer.trainable}")

# =============================================================================
# SECTION 10: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. DENSE = FULLY CONNECTED
   - Each input connects to each output
   - Parameters: (input_features x units) + units

2. WEIGHT ACCESS:
   - get_weights() returns [W, b]
   - set_weights([W, b]) to set custom values
   - kernel = W, bias = b (Keras terminology)

3. KERAS MATCHES MANUAL MATH:
   Dense layer computes: activation(W @ x + b)
   Exactly what we did yesterday!

4. INITIALIZATION MATTERS:
   - 'glorot_uniform': Default, good general choice
   - 'he_normal': Better for ReLU activations
   - Bad initialization = hard to train

5. DEBUG TIP:
   - Use model.summary() to see shapes
   - Use get_weights() to inspect values
   - Compare manual vs Keras to verify understanding

TOMORROW: Convolutional layers for images!
""")

