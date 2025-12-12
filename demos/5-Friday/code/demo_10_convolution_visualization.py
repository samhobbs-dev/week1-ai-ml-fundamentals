"""
Demo 10: Convolution Visualization
==================================
Week 1, Friday - CNN and Training Visualization

Visualize the convolution operation: filters, feature maps, edge detection.
Trainees will SEE what convolutions actually do.

INSTRUCTOR NOTES:
- Start with the convolution diagram
- Show edge detection filters - very visual and intuitive
- This explains WHY CNNs work for images

Estimated Time: 20-25 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# SECTION 1: INTRODUCTION
# =============================================================================
print("=" * 60)
print("DEMO 10: CONVOLUTION VISUALIZATION")
print("Seeing What CNNs Actually See")
print("=" * 60)

print("""
CONVOLUTION: A filter slides across an image, detecting patterns.

Think of it like a flashlight:
- The filter is the flashlight's pattern
- It shines on each part of the image
- High response = pattern found!
""")

# =============================================================================
# SECTION 2: CREATE A SIMPLE IMAGE
# =============================================================================
print("\n--- CREATING TEST IMAGE ---")

# Create a 7x7 image with a vertical edge
image = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.float32)

print("Image with vertical edge (0=dark, 1=light):")
print(image)

plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.title('Test Image: Vertical Edge')
plt.colorbar()
plt.tight_layout()
plt.savefig('10_test_image.png', dpi=100)
plt.show()
print("[Saved: 10_test_image.png]")

# =============================================================================
# SECTION 3: EDGE DETECTION FILTERS
# =============================================================================
print("\n--- EDGE DETECTION FILTERS ---")

# Vertical edge detector
vertical_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# Horizontal edge detector
horizontal_filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# Sobel filters (more sophisticated)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

print("Vertical Edge Filter:")
print(vertical_filter)
print("\nHorizontal Edge Filter:")
print(horizontal_filter)

# =============================================================================
# SECTION 4: MANUAL CONVOLUTION
# =============================================================================
print("\n--- MANUAL CONVOLUTION ---")

def convolve2d(image, kernel):
    """Perform 2D convolution."""
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Apply vertical edge filter
vertical_output = convolve2d(image, vertical_filter)

print("After applying vertical edge filter:")
print(vertical_output)

print("\nNotice: High values where the vertical edge is!")

# =============================================================================
# SECTION 5: VISUALIZE FILTER RESULTS
# =============================================================================
print("\n--- VISUALIZING FILTER RESULTS ---")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')

# Vertical filter
axes[0, 1].imshow(vertical_filter, cmap='RdBu', vmin=-1, vmax=1)
axes[0, 1].set_title('Vertical Edge Filter')
for i in range(3):
    for j in range(3):
        axes[0, 1].text(j, i, f'{vertical_filter[i,j]:.0f}', ha='center', va='center')

# Vertical filter result
axes[0, 2].imshow(vertical_output, cmap='gray')
axes[0, 2].set_title('Vertical Edge Detected!')

# Create image with horizontal edge
image_h = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.float32)

# Horizontal edge result
horizontal_output = convolve2d(image_h, horizontal_filter)

axes[1, 0].imshow(image_h, cmap='gray')
axes[1, 0].set_title('Image with Horizontal Edge')

axes[1, 1].imshow(horizontal_filter, cmap='RdBu', vmin=-1, vmax=1)
axes[1, 1].set_title('Horizontal Edge Filter')
for i in range(3):
    for j in range(3):
        axes[1, 1].text(j, i, f'{horizontal_filter[i,j]:.0f}', ha='center', va='center')

axes[1, 2].imshow(horizontal_output, cmap='gray')
axes[1, 2].set_title('Horizontal Edge Detected!')

plt.tight_layout()
plt.savefig('10_edge_detection.png', dpi=100)
plt.show()
print("[Saved: 10_edge_detection.png]")

# =============================================================================
# SECTION 6: KERAS CONV2D
# =============================================================================
print("\n--- KERAS CONV2D ---")

# Reshape for Keras: (batch, height, width, channels)
image_keras = image.reshape(1, 7, 7, 1)

# Create Conv2D layer with custom filter
conv_layer = layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding='valid',
    use_bias=False,
    input_shape=(7, 7, 1)
)

# Build and set custom weights
conv_layer.build((None, 7, 7, 1))
conv_layer.set_weights([vertical_filter.reshape(3, 3, 1, 1)])

# Apply convolution
keras_output = conv_layer(image_keras).numpy()

print("Keras Conv2D output:")
print(keras_output.reshape(5, 5))
print("\nMatches manual convolution:", np.allclose(vertical_output, keras_output.reshape(5, 5)))

# =============================================================================
# SECTION 7: MULTIPLE FILTERS
# =============================================================================
print("\n--- MULTIPLE FILTERS ---")

# Create Conv2D with multiple filters
multi_conv = layers.Conv2D(
    filters=4,  # 4 different filters
    kernel_size=3,
    padding='valid',
    use_bias=False,
    input_shape=(7, 7, 1)
)

# Build and set filters
multi_conv.build((None, 7, 7, 1))

# Stack different filters
filters = np.stack([
    vertical_filter,
    horizontal_filter,
    sobel_x,
    sobel_y
], axis=-1).reshape(3, 3, 1, 4)

multi_conv.set_weights([filters])

# Apply
multi_output = multi_conv(image_keras).numpy()

print(f"Input shape: {image_keras.shape}")
print(f"Output shape: {multi_output.shape}")
print("4 filters produce 4 feature maps!")

# Visualize all feature maps
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input')

filter_names = ['Vertical', 'Horizontal', 'Sobel X', 'Sobel Y']
for i, name in enumerate(filter_names):
    axes[i+1].imshow(multi_output[0, :, :, i], cmap='gray')
    axes[i+1].set_title(f'{name} Filter')

plt.tight_layout()
plt.savefig('10_multiple_filters.png', dpi=100)
plt.show()
print("[Saved: 10_multiple_filters.png]")

# =============================================================================
# SECTION 8: REAL IMAGE EXAMPLE
# =============================================================================
print("\n--- REAL IMAGE EXAMPLE ---")

# Create a more complex image
np.random.seed(42)
real_image = np.zeros((28, 28), dtype=np.float32)

# Add shapes
real_image[5:15, 10:20] = 1.0  # Square
real_image[20:25, 5:25] = 0.5  # Rectangle

# Add some noise
real_image += np.random.normal(0, 0.1, (28, 28))
real_image = np.clip(real_image, 0, 1)

# Apply filters
real_keras = real_image.reshape(1, 28, 28, 1)

# Create fresh conv layer
edge_conv = layers.Conv2D(2, 3, padding='same', use_bias=False, input_shape=(28, 28, 1))
edge_conv.build((None, 28, 28, 1))

# Set vertical and horizontal filters
edge_filters = np.stack([vertical_filter, horizontal_filter], axis=-1).reshape(3, 3, 1, 2)
edge_conv.set_weights([edge_filters])

# Apply
real_output = edge_conv(real_keras).numpy()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(real_image, cmap='gray')
axes[0].set_title('Input Image')

axes[1].imshow(real_output[0, :, :, 0], cmap='gray')
axes[1].set_title('Vertical Edges')

axes[2].imshow(real_output[0, :, :, 1], cmap='gray')
axes[2].set_title('Horizontal Edges')

plt.tight_layout()
plt.savefig('10_real_image_edges.png', dpi=100)
plt.show()
print("[Saved: 10_real_image_edges.png]")

# =============================================================================
# SECTION 9: LEARNED FILTERS
# =============================================================================
print("\n--- LEARNED FILTERS ---")
print("In a trained CNN, filters are LEARNED, not hand-crafted!")

# Create a small CNN
cnn = keras.Sequential([
    layers.Conv2D(8, 3, activation='relu', input_shape=(28, 28, 1)),
])

# Random weights (untrained)
print("\nUntrained filter shapes:")
print(f"Kernel shape: {cnn.layers[0].kernel.shape}")
print("(3x3 filter, 1 input channel, 8 output filters)")

# Get filters
filters_random = cnn.layers[0].get_weights()[0]

# Visualize first 4 filters
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    ax = axes[i // 4, i % 4]
    ax.imshow(filters_random[:, :, 0, i], cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title(f'Filter {i+1}')
    ax.axis('off')

plt.suptitle('Random (Untrained) Filters', fontsize=14)
plt.tight_layout()
plt.savefig('10_random_filters.png', dpi=100)
plt.show()
print("[Saved: 10_random_filters.png]")

print("""
After training:
- Early layer filters learn edges, colors, textures
- Later layer filters learn complex patterns
- The network discovers what features matter!
""")

# =============================================================================
# SECTION 10: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. CONVOLUTION = PATTERN MATCHING
   - Filter slides across image
   - High response where pattern matches
   - Same filter works everywhere (weight sharing!)

2. EDGE DETECTION FILTERS:
   - Vertical filter: [-1, 0, 1] pattern
   - Horizontal filter: [-1; 0; 1] pattern
   - These are classic, hand-crafted examples

3. FEATURE MAPS:
   - Each filter produces one feature map
   - 32 filters = 32 feature maps
   - Later layers combine these features

4. LEARNED FILTERS:
   - CNN filters are LEARNED from data
   - Early layers: edges, textures
   - Later layers: complex patterns, shapes

5. OUTPUT SIZE:
   - valid padding: (input - filter + 1)
   - same padding: same as input

NEXT: Building a complete CNN architecture!
""")

