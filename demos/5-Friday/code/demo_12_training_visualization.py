"""
Demo 12: Training Visualization
==============================
Week 1, Friday - CNN and Training Visualization

Visualize training metrics to understand model learning.
Detect overfitting, underfitting, and when to stop training.

INSTRUCTOR NOTES:
- This is crucial for practical ML work
- Show both good and bad training curves
- Introduce early stopping as a practical tool

Estimated Time: 15-20 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

# =============================================================================
# SECTION 1: INTRODUCTION
# =============================================================================
print("=" * 60)
print("DEMO 12: TRAINING VISUALIZATION")
print("Understanding How Your Model Learns")
print("=" * 60)

print("""
Training without visualization is like driving blindfolded.

We need to see:
- Is the model learning? (loss decreasing)
- Is it overfitting? (val loss increasing)
- When should we stop? (early stopping)
""")

# =============================================================================
# SECTION 2: PREPARE DATA
# =============================================================================
print("\n--- PREPARING DATA ---")

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Use subset for faster demo
X_train_sub = X_train[:5000]
y_train_sub = y_train[:5000]
X_val = X_train[5000:6000]
y_val = y_train[5000:6000]

print(f"Training samples: {len(X_train_sub)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# =============================================================================
# SECTION 3: BUILD MODEL
# =============================================================================
print("\n--- BUILDING MODEL ---")

def create_cnn():
    """Create a simple CNN."""
    return keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

model = create_cnn()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created and compiled!")

# =============================================================================
# SECTION 4: TRAIN WITH HISTORY
# =============================================================================
print("\n--- TRAINING WITH HISTORY ---")

history = model.fit(
    X_train_sub, y_train_sub,
    epochs=15,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

# =============================================================================
# SECTION 5: PLOT TRAINING CURVES
# =============================================================================
print("\n--- PLOTTING TRAINING CURVES ---")

def plot_training_history(history, title_suffix=''):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title(f'Model Loss {title_suffix}', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title(f'Model Accuracy {title_suffix}', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

fig = plot_training_history(history, '(Normal Training)')
plt.savefig('12_training_curves.png', dpi=100)
plt.show()
print("[Saved: 12_training_curves.png]")

# =============================================================================
# SECTION 6: INTERPRETING CURVES
# =============================================================================
print("\n--- INTERPRETING THE CURVES ---")

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Training Accuracy: {final_train_acc:.2%}")
print(f"Final Validation Accuracy: {final_val_acc:.2%}")

gap = final_train_acc - final_val_acc
print(f"\nAccuracy Gap (train - val): {gap:.2%}")

if gap > 0.05:
    print("Warning: Possible OVERFITTING (training >> validation)")
elif final_val_acc < 0.8:
    print("Warning: Possible UNDERFITTING (low accuracy)")
else:
    print("Training looks healthy!")

# =============================================================================
# SECTION 7: DEMONSTRATE OVERFITTING
# =============================================================================
print("\n--- DEMONSTRATING OVERFITTING ---")
print("Training a model that will overfit...")

# Create an overfitting-prone model (too complex for the data)
model_overfit = keras.Sequential([
    layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_overfit.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train on very small subset
history_overfit = model_overfit.fit(
    X_train_sub[:500], y_train_sub[:500],  # Small dataset
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

fig = plot_training_history(history_overfit, '(Overfitting Example)')
plt.savefig('12_overfitting.png', dpi=100)
plt.show()
print("[Saved: 12_overfitting.png]")

print("""
OVERFITTING SIGNS:
- Training loss keeps decreasing
- Validation loss starts INCREASING
- Training accuracy >> Validation accuracy
""")

# =============================================================================
# SECTION 8: EARLY STOPPING
# =============================================================================
print("\n--- EARLY STOPPING ---")
print("The solution: stop training when validation stops improving!")

# Create fresh model
model_early = create_cnn()
model_early.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining with early stopping (patience=3)...")
history_early = model_early.fit(
    X_train_sub, y_train_sub,
    epochs=50,  # Set high, early stopping will stop
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

print(f"\nStopped at epoch: {len(history_early.history['loss'])}")
best_epoch = np.argmin(history_early.history['val_loss']) + 1
print(f"Best epoch (lowest val_loss): {best_epoch}")

fig = plot_training_history(history_early, '(With Early Stopping)')
plt.axvline(x=best_epoch-1, color='green', linestyle='--', label='Best Model')
plt.legend()
plt.savefig('12_early_stopping.png', dpi=100)
plt.show()
print("[Saved: 12_early_stopping.png]")

# =============================================================================
# SECTION 9: MODEL CHECKPOINTING
# =============================================================================
print("\n--- MODEL CHECKPOINTING ---")
print("Save the best model during training!")

# Create fresh model
model_checkpoint = create_cnn()
model_checkpoint.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("\nTraining with checkpointing...")
history_ckpt = model_checkpoint.fit(
    X_train_sub, y_train_sub,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop],
    verbose=0
)

print(f"\nTraining complete! Best model saved to 'best_model.keras'")

# =============================================================================
# SECTION 10: COMPARING TRAINING RUNS
# =============================================================================
print("\n--- COMPARING TRAINING RUNS ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Compare validation losses
axes[0].plot(history.history['val_loss'], label='Normal', linewidth=2)
axes[0].plot(history_early.history['val_loss'], label='With Early Stopping', linewidth=2)
axes[0].set_title('Validation Loss Comparison', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Compare validation accuracies
axes[1].plot(history.history['val_accuracy'], label='Normal', linewidth=2)
axes[1].plot(history_early.history['val_accuracy'], label='With Early Stopping', linewidth=2)
axes[1].set_title('Validation Accuracy Comparison', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('12_comparison.png', dpi=100)
plt.show()
print("[Saved: 12_comparison.png]")

# =============================================================================
# SECTION 11: TRAINING CURVE DIAGNOSIS
# =============================================================================
print("\n--- TRAINING CURVE DIAGNOSIS ---")

print("""
+------------------+---------------------------+------------------------+
| Pattern          | Diagnosis                 | Solution               |
+------------------+---------------------------+------------------------+
| Both losses high | UNDERFITTING              | More capacity, longer  |
|                  |                           | training, less regulz  |
+------------------+---------------------------+------------------------+
| Train << Val     | OVERFITTING               | More data, dropout,    |
|                  |                           | early stopping, regulz |
+------------------+---------------------------+------------------------+
| Val increasing   | OVERFITTING (severe)      | Stop training! Add     |
|                  |                           | regularization         |
+------------------+---------------------------+------------------------+
| Both converging  | GOOD FIT                  | Try more epochs or     |
|                  |                           | slight tuning          |
+------------------+---------------------------+------------------------+
| Loss oscillating | Learning rate too high    | Reduce learning rate   |
+------------------+---------------------------+------------------------+
| Loss stuck       | Learning rate too low     | Increase learning rate |
+------------------+---------------------------+------------------------+
""")

# =============================================================================
# SECTION 12: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. ALWAYS VISUALIZE TRAINING:
   - Plot loss and accuracy curves
   - Compare training vs validation

2. OVERFITTING = MEMORIZATION:
   - Training improves, validation worsens
   - Fix: More data, dropout, early stopping

3. UNDERFITTING = CAN'T LEARN:
   - Both metrics are poor
   - Fix: More capacity, longer training

4. EARLY STOPPING:
   - Automatically stop when val_loss stops improving
   - Set patience (epochs to wait)
   - restore_best_weights=True

5. MODEL CHECKPOINTING:
   - Save best model during training
   - Never lose your best results!

CONGRATULATIONS! YOU'VE COMPLETED WEEK 1!
From ML basics to building and visualizing CNNs.
""")

print("\n" + "=" * 60)
print("WEEK 1 COMPLETE: FROM ZERO TO NEURAL!")
print("=" * 60)
print("""
Your journey this week:
- Tuesday: ML fundamentals (regression, classification, clustering)
- Wednesday: Neural network architecture and forward propagation
- Thursday: TensorFlow and Keras implementation
- Friday: CNN architecture and training visualization

Next week: Deep dive into backpropagation and advanced training!
""")

