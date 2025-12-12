"""
Demo 11: Building a Complete CNN Architecture
=============================================
Week 1, Friday - CNN and Training Visualization

Build a complete CNN for image classification.
Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Output

INSTRUCTOR NOTES:
- Have the CNN architecture diagram visible
- Trace shapes through each layer
- This is the capstone of Week 1!

Estimated Time: 20-25 minutes
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# SECTION 1: INTRODUCTION
# =============================================================================
print("=" * 60)
print("DEMO 11: BUILDING A COMPLETE CNN")
print("The Culmination of Week 1!")
print("=" * 60)

print("""
CNN ARCHITECTURE PATTERN:

[Conv -> ReLU -> Pool] x N  ->  [Flatten]  ->  [Dense] x M  ->  [Output]
    Feature Extraction            Bridge         Classification

Today we build this from scratch!
""")

# =============================================================================
# SECTION 2: LOAD MNIST DATA
# =============================================================================
print("\n--- LOADING MNIST DATA ---")

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training images: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test images: {X_test.shape}")
print(f"Pixel range: [{X_train.min()}, {X_train.max()}]")

# Preprocessing
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add channel dimension for Conv2D: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"\nAfter preprocessing:")
print(f"Training shape: {X_train.shape}")
print(f"Pixel range: [{X_train.min()}, {X_train.max()}]")

# =============================================================================
# SECTION 3: BUILD THE CNN
# =============================================================================
print("\n--- BUILDING THE CNN ---")
print("Following the architecture diagram step by step:")

model = keras.Sequential([
    # Convolutional Block 1
    # Input: (28, 28, 1)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    # Output: (26, 26, 32)
    layers.MaxPooling2D((2, 2), name='pool1'),
    # Output: (13, 13, 32)
    
    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    # Output: (11, 11, 64)
    layers.MaxPooling2D((2, 2), name='pool2'),
    # Output: (5, 5, 64)
    
    # Convolutional Block 3
    layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
    # Output: (3, 3, 64)
    
    # Classification Head
    layers.Flatten(name='flatten'),
    # Output: (576)
    layers.Dense(64, activation='relu', name='dense1'),
    # Output: (64)
    layers.Dense(10, activation='softmax', name='output')
    # Output: (10)
], name='mnist_cnn')

print("\nModel created!")

# =============================================================================
# SECTION 4: MODEL SUMMARY
# =============================================================================
print("\n--- MODEL SUMMARY ---")
model.summary()

# =============================================================================
# SECTION 5: TRACE SHAPES
# =============================================================================
print("\n--- SHAPE TRACING ---")

# Create a sample input
sample = X_train[:1]
print(f"Input shape: {sample.shape}")

print("\nLayer-by-layer shapes:")
x = sample
for layer in model.layers:
    x = layer(x)
    params = layer.count_params()
    print(f"  {layer.name:<10}: {str(x.shape):<20} params: {params}")

# =============================================================================
# SECTION 6: PARAMETER BREAKDOWN
# =============================================================================
print("\n--- PARAMETER COUNT BREAKDOWN ---")

total_params = 0
for layer in model.layers:
    params = layer.count_params()
    total_params += params
    
    if 'conv' in layer.name:
        # Conv2D params = (kernel_h * kernel_w * in_channels + 1) * out_channels
        weights = layer.get_weights()
        if weights:
            W = weights[0]
            print(f"{layer.name}: {W.shape} -> ({W.shape[0]}*{W.shape[1]}*{W.shape[2]}+1)*{W.shape[3]} = {params}")
    elif 'dense' in layer.name or 'output' in layer.name:
        weights = layer.get_weights()
        if weights:
            W = weights[0]
            print(f"{layer.name}: ({W.shape[0]}+1)*{W.shape[1]} = {params}")

print(f"\nTotal trainable parameters: {total_params:,}")

# =============================================================================
# SECTION 7: COMPILE THE MODEL
# =============================================================================
print("\n--- COMPILING MODEL ---")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Compiled with:")
print("  Optimizer: Adam")
print("  Loss: Sparse Categorical Crossentropy")
print("  Metrics: Accuracy")

# =============================================================================
# SECTION 8: TRAIN THE MODEL (BRIEF)
# =============================================================================
print("\n--- TRAINING (Brief Demo) ---")
print("Training for 3 epochs to show the process...")
print("(Full training would use more epochs)")

# Use subset for faster demo
history = model.fit(
    X_train[:10000], y_train[:10000],
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# =============================================================================
# SECTION 9: EVALUATE
# =============================================================================
print("\n--- EVALUATION ---")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")

print("\nNote: With more epochs and full data, accuracy > 99% is achievable!")

# =============================================================================
# SECTION 10: MAKE PREDICTIONS
# =============================================================================
print("\n--- PREDICTIONS ---")

# Predict on first 10 test images
predictions = model.predict(X_test[:10], verbose=0)

print("First 10 predictions:")
print(f"{'Index':<8}{'Predicted':<12}{'Actual':<12}{'Confidence':<12}{'Status'}")
print("-" * 50)

for i in range(10):
    pred_class = np.argmax(predictions[i])
    confidence = predictions[i][pred_class]
    actual = y_test[i]
    status = "Correct" if pred_class == actual else "WRONG"
    print(f"{i:<8}{pred_class:<12}{actual:<12}{confidence:.2%}{'':>4}{status}")

# =============================================================================
# SECTION 11: VISUALIZE FEATURE MAPS
# =============================================================================
print("\n--- FEATURE MAPS VISUALIZATION ---")

import matplotlib.pyplot as plt

# Get intermediate outputs
feature_model = keras.Model(
    inputs=model.inputs,
    outputs=[model.get_layer('conv1').output,
             model.get_layer('conv2').output,
             model.get_layer('conv3').output]
)

# Get feature maps for one image
sample_image = X_test[0:1]
feature_maps = feature_model.predict(sample_image, verbose=0)

# Plot
fig, axes = plt.subplots(4, 8, figsize=(16, 8))

# Original image
axes[0, 0].imshow(sample_image[0, :, :, 0], cmap='gray')
axes[0, 0].set_title('Input')
axes[0, 0].axis('off')

# Hide unused plots in first row
for j in range(1, 8):
    axes[0, j].axis('off')

# Conv1 feature maps (first 8)
for i in range(8):
    axes[1, i].imshow(feature_maps[0][0, :, :, i], cmap='viridis')
    axes[1, i].axis('off')
axes[1, 0].set_ylabel('Conv1', fontsize=10)

# Conv2 feature maps (first 8)
for i in range(8):
    axes[2, i].imshow(feature_maps[1][0, :, :, i], cmap='viridis')
    axes[2, i].axis('off')
axes[2, 0].set_ylabel('Conv2', fontsize=10)

# Conv3 feature maps (first 8)
for i in range(8):
    axes[3, i].imshow(feature_maps[2][0, :, :, i], cmap='viridis')
    axes[3, i].axis('off')
axes[3, 0].set_ylabel('Conv3', fontsize=10)

plt.suptitle('Feature Maps Through CNN Layers', fontsize=14)
plt.tight_layout()
plt.savefig('11_feature_maps.png', dpi=100)
plt.show()
print("[Saved: 11_feature_maps.png]")

# =============================================================================
# SECTION 12: CNN ARCHITECTURE VARIATIONS
# =============================================================================
print("\n--- ARCHITECTURE VARIATIONS ---")

print("""
Common CNN patterns:

1. BASIC (what we built):
   Conv -> Pool -> Conv -> Pool -> Dense

2. VGG-STYLE (deeper):
   Conv -> Conv -> Pool -> Conv -> Conv -> Pool -> Dense

3. WITH DROPOUT (regularization):
   Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Dropout -> Dense

4. WITH BATCH NORMALIZATION (faster training):
   Conv -> BatchNorm -> ReLU -> Pool -> Conv -> BatchNorm -> ReLU -> Pool
""")

# Example with dropout
model_dropout = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("\nModel with Dropout:")
model_dropout.summary()

# =============================================================================
# SECTION 13: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. CNN ARCHITECTURE:
   [Conv Blocks] -> [Flatten] -> [Dense Layers] -> [Output]
   
2. CONV BLOCK PATTERN:
   Conv2D -> ReLU -> MaxPooling
   
3. SHAPE PROGRESSION:
   (28,28,1) -> (13,13,32) -> (5,5,64) -> (3,3,64) -> (576) -> (10)
   Spatial dims decrease, channels increase, then flatten!

4. PARAMETER EFFICIENCY:
   Conv layers have few parameters (weight sharing)
   Dense layers have many (especially after flatten)

5. FEATURE MAPS:
   Early layers: edges, simple patterns
   Later layers: complex, abstract features

6. TRAINING:
   Same workflow: compile -> fit -> evaluate -> predict

YOU'VE BUILT YOUR FIRST CNN!
Next demo: Visualizing training to understand learning.
""")

