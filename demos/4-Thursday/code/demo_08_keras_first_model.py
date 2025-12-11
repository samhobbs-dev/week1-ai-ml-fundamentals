"""
Demo 08: Building Your First Keras Model
=========================================
Week 1, Thursday - TensorFlow and Keras

Building neural networks with Keras Sequential API.
This is the moment trainees become capable of building real models!

INSTRUCTOR NOTES:
- This is the "payoff" for all the theory
- Show how concepts map: Dense(64) = 64 neurons
- The model summary is a key skill

Estimated Time: 20-25 minutes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# =============================================================================
# SECTION 1: INTRODUCTION
# =============================================================================
print("=" * 60)
print("DEMO 08: YOUR FIRST KERAS MODEL")
print("From Theory to Working Code")
print("=" * 60)

print("""
KERAS is TensorFlow's high-level API.
It turns neural network theory into practical, readable code.

What took 100+ lines of NumPy yesterday...
...takes ~10 lines of Keras today!
""")

# =============================================================================
# SECTION 2: THE SEQUENTIAL API
# =============================================================================
print("\n--- THE SEQUENTIAL API ---")
print("Sequential = layers stacked one after another")
print("Input -> Layer 1 -> Layer 2 -> ... -> Output")

# Create a simple model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("\nModel created!")
print("Architecture: 784 -> 64 -> 32 -> 10")

# =============================================================================
# SECTION 3: MODEL SUMMARY
# =============================================================================
print("\n--- MODEL SUMMARY ---")
print("This is a KEY SKILL: reading model summaries!\n")

model.summary()

print("""
READING THE SUMMARY:
- Layer name and type
- Output Shape: (None, units) where None = batch size
- Param #: (input_features x units) + units (weights + biases)

Parameter count for each layer:
  Layer 1: (784 x 64) + 64 = 50,240
  Layer 2: (64 x 32) + 32 = 2,080  
  Layer 3: (32 x 10) + 10 = 330
  Total: 52,650 trainable parameters
""")

# =============================================================================
# SECTION 4: BUILDING STEP BY STEP
# =============================================================================
print("\n--- BUILDING STEP BY STEP ---")

# Method 2: Add layers one at a time
model2 = keras.Sequential(name='step_by_step_model')
model2.add(layers.Dense(128, activation='relu', input_shape=(784,), name='hidden_1'))
model2.add(layers.Dense(64, activation='relu', name='hidden_2'))
model2.add(layers.Dense(10, activation='softmax', name='output'))

print("Model built layer by layer:")
model2.summary()

# =============================================================================
# SECTION 5: COMPILING THE MODEL
# =============================================================================
print("\n--- COMPILING THE MODEL ---")
print("Compile = set up the training configuration")

model.compile(
    optimizer='adam',           # How to update weights
    loss='categorical_crossentropy',  # What to minimize
    metrics=['accuracy']        # What to track
)

print("""
Compile settings:
- optimizer='adam': Adaptive learning rate optimizer (default choice)
- loss='categorical_crossentropy': For multi-class classification
- metrics=['accuracy']: Track classification accuracy

COMMON CONFIGURATIONS:
+-------------------+------------------------+------------------+
| Task              | Loss                   | Output Activation|
+-------------------+------------------------+------------------+
| Binary class      | binary_crossentropy    | sigmoid          |
| Multi-class       | categorical_crossentropy| softmax         |
| Multi-class (int) | sparse_categorical_ce  | softmax          |
| Regression        | mse                    | linear (none)    |
+-------------------+------------------------+------------------+
""")

# =============================================================================
# SECTION 6: DUMMY DATA TRAINING
# =============================================================================
print("\n--- TRAINING THE MODEL ---")

# Create dummy data (random, won't learn meaningful patterns)
np.random.seed(42)
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = keras.utils.to_categorical(np.random.randint(0, 10, 1000), 10)

X_val = np.random.randn(200, 784).astype(np.float32)
y_val = keras.utils.to_categorical(np.random.randint(0, 10, 200), 10)

print(f"Training data: X={X_train.shape}, y={y_train.shape}")
print(f"Validation data: X={X_val.shape}, y={y_val.shape}")

# Train the model
print("\nTraining for 5 epochs...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# =============================================================================
# SECTION 7: UNDERSTANDING TRAINING OUTPUT
# =============================================================================
print("\n--- UNDERSTANDING TRAINING OUTPUT ---")
print("""
Training output explained:
- 32/32 [====]: Progress through batches (1000 samples / 32 batch_size = 32 batches)
- loss: Training loss (decreasing = learning)
- accuracy: Training accuracy
- val_loss: Validation loss (test set)
- val_accuracy: Validation accuracy (what matters!)

Note: Random data won't show meaningful accuracy.
Real data (like MNIST tomorrow) will show much better results!
""")

# =============================================================================
# SECTION 8: MAKING PREDICTIONS
# =============================================================================
print("\n--- MAKING PREDICTIONS ---")

# Predict on validation data
predictions = model.predict(X_val[:5], verbose=0)

print("Predictions for first 5 samples:")
print(f"Raw output shape: {predictions.shape}")
print("\nProbabilities (one row per sample, 10 classes):")
for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class]
    print(f"  Sample {i+1}: Class {predicted_class} ({confidence:.2%} confidence)")

# =============================================================================
# SECTION 9: ACCESSING TRAINING HISTORY
# =============================================================================
print("\n--- TRAINING HISTORY ---")

print("History keys:", list(history.history.keys()))
print(f"\nLoss progression: {[f'{l:.4f}' for l in history.history['loss']]}")
print(f"Accuracy progression: {[f'{a:.2%}' for a in history.history['accuracy']]}")

# =============================================================================
# SECTION 10: DIFFERENT MODEL ARCHITECTURES
# =============================================================================
print("\n--- COMMON ARCHITECTURES ---")

# Binary Classification
print("\n1. Binary Classification:")
binary_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Single output, sigmoid
])
binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("   Output: 1 neuron with sigmoid (probability 0-1)")

# Multi-class Classification
print("\n2. Multi-class Classification (10 classes):")
multiclass_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 outputs, softmax for probability distribution
])
multiclass_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("   Output: 10 neurons with softmax (probability distribution)")
print("   Note: Softmax is special for output layers - converts scores to probabilities that sum to 1")

# Regression
print("\n3. Regression:")
regression_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(13,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation for regression!
])
regression_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("   Output: 1 neuron with linear activation (any real number)")

# =============================================================================
# SECTION 11: EVALUATING THE MODEL
# =============================================================================
print("\n--- MODEL EVALUATION ---")

loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.2%}")

# =============================================================================
# SECTION 12: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. SEQUENTIAL API: Stack layers linearly
   model = keras.Sequential([layer1, layer2, layer3])

2. DENSE LAYERS: Fully connected neurons
   layers.Dense(units, activation)
   - units: number of neurons
   - activation: 'relu', 'sigmoid', 'softmax', or None

3. COMPILE: Set up training configuration
   model.compile(optimizer, loss, metrics)

4. FIT: Train the model
   model.fit(X, y, epochs, batch_size, validation_data)

5. PREDICT: Make predictions
   model.predict(X_new)

6. EVALUATE: Test performance
   model.evaluate(X_test, y_test)

THE KERAS WORKFLOW:
Define -> Compile -> Fit -> Evaluate -> Predict

NEXT: Looking inside Dense layers!
""")

