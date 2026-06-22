"""
================================================================================
  DATA ANALYSIS ASSIGNMENT - 5th Exercise
  Aristotle University of Thessaloniki - Mechanical Engineering
  Academic Year 2025-26 | Semester 8
  Subject: Neural Networks & CNN for MNIST Digit Classification
  AEM (seed): 7137
================================================================================

INSTRUCTIONS FOR USE IN VS CODE:
  1. Place this script in the same directory as your zip files:
       your_folder/
         NN_2.py
         train_dataset.zip
         test_dataset.zip
  2. Install dependencies:
       pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
  3. Run: python NN_3.py
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import zipfile

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

AEM        = 7137    # Student ID used as random seed
EPOCHS     = 30
VAL_SPLIT  = 0.10
BATCH_SIZE = 64

# ------------------------------------------------------------------------------
# AUTO-UNZIP
# Extracts train/test zips if the folders do not already exist.
# Works even if the zip created a nested subfolder inside.
# ------------------------------------------------------------------------------

for zip_name, folder_name in [("train_dataset.zip", "train_dataset"),
                               ("test_dataset.zip",  "test_dataset")]:
    if not os.path.exists(folder_name):
        if os.path.exists(zip_name):
            print(f"  Extracting {zip_name}...")
            with zipfile.ZipFile(zip_name, "r") as z:
                z.extractall(".")
            print(f"  Done -> {folder_name}/")
        else:
            raise FileNotFoundError(
                f"Neither '{folder_name}/' nor '{zip_name}' found. "
                f"Make sure the zip file is in the same folder as this script."
            )
    else:
        print(f"  {folder_name}/ already exists, skipping extraction.")


# ------------------------------------------------------------------------------
# PATH DETECTION
# Finds the actual folder containing labels.csv, even if the zip created
# a nested subfolder (e.g. test_dataset/test_dataset/labels.csv).
# ------------------------------------------------------------------------------

def find_dataset_root(base_folder):
    for root, dirs, files in os.walk(base_folder):
        if "labels.csv" in files:
            return root
    raise FileNotFoundError(
        f"Could not find labels.csv anywhere inside '{base_folder}/'. "
        f"Check that the zip extracted correctly."
    )

# DEBUG - find where images actually are
import glob
pngs = glob.glob("**/*.png", recursive=True)
print("First 10 PNG files found:")
for p in pngs[:10]:
    print(" ", p)
    
TRAIN_ROOT = find_dataset_root("train_dataset")
TEST_ROOT  = find_dataset_root("test_dataset")
TRAIN_CSV  = os.path.join(TRAIN_ROOT, "labels.csv")
TEST_CSV   = os.path.join(TEST_ROOT,  "labels.csv")

print(f"  Train root : {TRAIN_ROOT}")
print(f"  Test  root : {TEST_ROOT}")


# ==============================================================================
# STEP 1 - LOAD CSV LABELS AND SUBSAMPLE WITH AEM SEED
# ==============================================================================

print("=" * 70)
print("STEP 1: Loading and subsampling data (seed = AEM = 7137)")
print("=" * 70)

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"  Original train set size : {len(train_df):,} samples")
print(f"  Original test  set size : {len(test_df):,} samples")

# Sample 90% of train, 80% of test using AEM as seed for reproducibility
train_df = train_df.sample(frac=0.90, random_state=AEM).reset_index(drop=True)
test_df  = test_df.sample(frac=0.80, random_state=AEM).reset_index(drop=True)

print(f"  Subsampled train set    : {len(train_df):,} samples (90%)")
print(f"  Subsampled test  set    : {len(test_df):,} samples (80%)")


# ------------------------------------------------------------------------------
# HELPER - Load images from disk into a NumPy array
# ------------------------------------------------------------------------------

def load_images(df, base_folder):
    """Read PNG files listed in df['file'] and return a (N, 28, 28) uint8 array."""
    images = []
    for path in df["file"]:
        full_path = os.path.join(base_folder, os.path.normpath(path))
        img = Image.open(full_path).convert("L")   # force grayscale
        images.append(np.array(img))
    return np.array(images, dtype=np.uint8)

print("\n  Loading train images from disk...")
X_train_raw = load_images(train_df, TRAIN_ROOT)
y_train     = train_df["label"].values

print("  Loading test images from disk...")
X_test_raw  = load_images(test_df, TEST_ROOT)
y_test      = test_df["label"].values


# ==============================================================================
# STEP 2 - EXPLORE DATA SHAPE AND CLASS BALANCE
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 2: Data shape & class balance")
print("=" * 70)

print(f"\n  X_train shape : {X_train_raw.shape}  (samples, height, width)")
print(f"  X_test  shape : {X_test_raw.shape}")
print(f"  Unique labels : {np.unique(y_train)}  (digits 0-9)\n")

unique, counts = np.unique(y_train, return_counts=True)
print("  Samples per digit in train set:")
for digit, count in zip(unique, counts):
    
    print(f"    Digit {digit}: {count:,}  ")

plt.figure(figsize=(10, 4))
plt.bar(unique, counts, color="steelblue", edgecolor="white")
plt.xticks(unique)
plt.xlabel("Digit Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Train Set")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
print("  -> Saved: class_distribution.png")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit in range(10):
    idx = np.where(y_train == digit)[0][0]
    ax = axes[digit // 5][digit % 5]
    ax.imshow(X_train_raw[idx], cmap="gray")
    ax.set_title(f"Digit: {digit}")
    ax.axis("off")
plt.suptitle("Sample Images per Digit Class")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.show()
print("  -> Saved: sample_images.png")


# ==============================================================================
# STEP 3 - NORMALIZATION  (pixel values -> [0, 1])
#
# WHY NORMALIZE?
#   Raw pixel values range from 0 to 255. Large input values lead to large
#   weight updates, making gradient descent unstable (exploding gradients).
#   Dividing by 255 maps all inputs to [0, 1], so every feature starts on
#   the same scale. This allows the optimizer (Adam) to converge faster and
#   more reliably, since the loss surface becomes smoother and more symmetric.
# ==============================================================================

X_train_norm = X_train_raw.astype("float32") / 255.0
X_test_norm  = X_test_raw.astype("float32")  / 255.0

print("\n" + "=" * 70)
print("STEP 3: Normalization complete  (pixel range -> [0.0, 1.0])")
print("=" * 70)
print(f"  Train pixel min: {X_train_norm.min():.1f}  max: {X_train_norm.max():.1f}")
print(f"  Test  pixel min: {X_test_norm.min():.1f}   max: {X_test_norm.max():.1f}")


# ==============================================================================
# STEP 4 - FLATTENING FOR THE SIMPLE NN  (28x28 -> 784)
#
# WHY FLATTEN?
#   A Dense (fully-connected) layer expects a 1D input vector. Each neuron
#   connects to every single input feature independently. By flattening we
#   give the layer the 784 pixel values as a single list. The CNN (Section B)
#   avoids this step until after convolutions have extracted spatial features.
# ==============================================================================

X_train_flat = X_train_norm.reshape(-1, 784)
X_test_flat  = X_test_norm.reshape(-1, 784)

print("\n" + "=" * 70)
print("STEP 4: Flattening complete  (28x28 -> 784 features per image)")
print("=" * 70)
print(f"  X_train_flat shape: {X_train_flat.shape}")
print(f"  X_test_flat  shape: {X_test_flat.shape}")


# ==============================================================================
# SECTION A - Simple Neural Network (Dense / MLP)
# ==============================================================================

print("\n\n" + "=" * 70)
print("SECTION A - Simple Neural Network (Dense NN)")
print("=" * 70)

# ------------------------------------------------------------------------------
# STEP 5 - BUILD THE SEQUENTIAL MODEL
#
# Architecture as specified in the assignment:
#   Input  Layer : Dense(784, relu)   - 784 neurons = one per pixel
#   Hidden Layer1: Dense(256, relu)   - captures mid-level patterns
#   Hidden Layer2: Dense(128, relu)   - compresses into higher abstractions
#   Output Layer : Dense(10, softmax) - probability for each digit (0-9)
#
# WHY ReLU?  f(x) = max(0, x) introduces non-linearity so the network can
#   learn curved decision boundaries. Avoids vanishing-gradient problem.
#
# WHY Softmax?  Converts 10 raw scores into a probability distribution
#   that sums to 1. The highest probability is the predicted digit.
# ------------------------------------------------------------------------------

model_nn = keras.Sequential([
    layers.Dense(784, activation="relu", input_shape=(784,)),  # Input layer
    layers.Dense(256, activation="relu"),                       # Hidden layer 1
    layers.Dense(128, activation="relu"),                       # Hidden layer 2
    layers.Dense(10,  activation="softmax"),                    # Output layer
], name="Simple_NN")

# ------------------------------------------------------------------------------
# STEP 6 - COMPILE
#
# Optimizer - Adam:
#   Adapts the learning rate individually for each parameter using estimates
#   of the first and second moments of gradients. Converges faster and more
#   robustly than vanilla SGD without manual learning-rate tuning.
#
# Loss - sparse_categorical_crossentropy:
#   For multi-class classification with integer labels (not one-hot encoded).
#   Penalises the model when it assigns low probability to the correct class.
# ------------------------------------------------------------------------------

model_nn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_nn.summary()

# STEP 7 - TRAIN

print("\n  Training Simple NN  (30 epochs, 10% validation split)...\n")

history_nn = model_nn.fit(
    X_train_flat, y_train,
    epochs=EPOCHS,
    validation_split=VAL_SPLIT,
    batch_size=BATCH_SIZE,
    verbose=1
)

# STEP 8 - EVALUATE: Accuracy & Loss Curves

test_loss_nn, test_acc_nn = model_nn.evaluate(X_test_flat, y_test, verbose=0)
print(f"\n  Simple NN - Test Accuracy : {test_acc_nn:.4f}")
print(f"  Simple NN - Test Loss     : {test_loss_nn:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_nn.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history_nn.history["val_accuracy"], label="Validation Accuracy", linestyle="--")
axes[0].set_title("Simple NN - Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_nn.history["loss"],     label="Train Loss")
axes[1].plot(history_nn.history["val_loss"], label="Validation Loss", linestyle="--")
axes[1].set_title("Simple NN - Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"Simple NN  |  Test Accuracy: {test_acc_nn:.4f}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("nn_training_curves.png", dpi=150)
plt.show()
print("  -> Saved: nn_training_curves.png")

# STEP 9 - CONFUSION MATRIX

y_pred_nn = np.argmax(model_nn.predict(X_test_flat, verbose=0), axis=1)
cm_nn     = confusion_matrix(y_test, y_pred_nn)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_nn, annot=True, fmt="d", cmap="Blues",
    xticklabels=range(10), yticklabels=range(10)
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Simple NN - Confusion Matrix  (Test Accuracy: {test_acc_nn:.4f})")
plt.tight_layout()
plt.savefig("nn_confusion_matrix.png", dpi=150)
plt.show()
print("  -> Saved: nn_confusion_matrix.png")

print("\n  Classification Report (Simple NN):")
print(classification_report(y_test, y_pred_nn, digits=4))

print("  Most confused digit pairs (off-diagonal):")
cm_copy = cm_nn.copy()
np.fill_diagonal(cm_copy, 0)
for _ in range(5):
    i, j = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    print(f"    True={i} -> Predicted={j}  ({cm_copy[i, j]} times)")
    cm_copy[i, j] = 0


# ==============================================================================
# SECTION B - Convolutional Neural Network (CNN)
# ==============================================================================

print("\n\n" + "=" * 70)
print("SECTION B - Convolutional Neural Network (CNN)")
print("=" * 70)

# ------------------------------------------------------------------------------
# STEP 10 - RESHAPE FOR CNN  (N, 28, 28) -> (N, 28, 28, 1)
#
# Conv2D layers expect 4D input: (batch, height, width, channels).
# MNIST is grayscale so channels = 1. We add this dimension explicitly.
# ------------------------------------------------------------------------------

X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
X_test_cnn  = X_test_norm.reshape(-1, 28, 28, 1)

print(f"\n  X_train_cnn shape: {X_train_cnn.shape}")
print(f"  X_test_cnn  shape: {X_test_cnn.shape}")

# ------------------------------------------------------------------------------
# STEP 11 - BUILD CNN MODEL
#
# Architecture (as specified in the assignment):
#   Conv2D(32, 3x3, relu)  - 32 filters detect simple features: edges, curves
#   MaxPooling2D(2x2)       - halves spatial dims; reduces computation
#   Conv2D(64, 3x3, relu)  - 64 filters combine edges into complex shapes
#   Flatten                 - unroll 3D feature maps into 1D for Dense layers
#   Dense(64, relu)         - classification reasoning layer
#   Dense(10, softmax)      - final digit probabilities
#
# WHY CNNs ARE BETTER FOR IMAGES:
#   - Weight sharing: a 3x3 filter applies the same weights across the whole
#     image -> far fewer parameters than a fully-connected approach.
#   - Translation invariance: the same feature is detected wherever it
#     appears in the image.
#   - Spatial hierarchy: shallow layers detect edges, deeper layers detect
#     complex structures like loops (6, 9) or angles (4, 7).
# ------------------------------------------------------------------------------

model_cnn = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64,  activation="relu"),
    layers.Dense(10,  activation="softmax"),
], name="CNN")

model_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_cnn.summary()

# STEP 12 - TRAIN CNN

print("\n  Training CNN  (30 epochs, 10% validation split)...\n")

history_cnn = model_cnn.fit(
    X_train_cnn, y_train,
    epochs=EPOCHS,
    validation_split=VAL_SPLIT,
    batch_size=BATCH_SIZE,
    verbose=1
)

# STEP 13 - EVALUATE CNN

test_loss_cnn, test_acc_cnn = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\n  CNN - Test Accuracy : {test_acc_cnn:.4f}")
print(f"  CNN - Test Loss     : {test_loss_cnn:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_cnn.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history_cnn.history["val_accuracy"], label="Validation Accuracy", linestyle="--")
axes[0].set_title("CNN - Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_cnn.history["loss"],     label="Train Loss")
axes[1].plot(history_cnn.history["val_loss"], label="Validation Loss", linestyle="--")
axes[1].set_title("CNN - Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"CNN  |  Test Accuracy: {test_acc_cnn:.4f}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=150)
plt.show()
print("  -> Saved: cnn_training_curves.png")

y_pred_cnn = np.argmax(model_cnn.predict(X_test_cnn, verbose=0), axis=1)
cm_cnn     = confusion_matrix(y_test, y_pred_cnn)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_cnn, annot=True, fmt="d", cmap="Greens",
    xticklabels=range(10), yticklabels=range(10)
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"CNN - Confusion Matrix  (Test Accuracy: {test_acc_cnn:.4f})")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png", dpi=150)
plt.show()
print("  -> Saved: cnn_confusion_matrix.png")

print("\n  Classification Report (CNN):")
print(classification_report(y_test, y_pred_cnn, digits=4))


# ==============================================================================
# FINAL COMPARISON - Simple NN vs CNN
# ==============================================================================

print("\n\n" + "=" * 70)
print("FINAL COMPARISON - Simple NN vs CNN")
print("=" * 70)

nn_params  = model_nn.count_params()
cnn_params = model_cnn.count_params()

print(f"\n  {'Metric':<30} {'Simple NN':>15} {'CNN':>15}")
print(f"  {'-'*60}")
print(f"  {'Test Accuracy':<30} {test_acc_nn:>14.4f} {test_acc_cnn:>14.4f}")
print(f"  {'Test Loss':<30} {test_loss_nn:>14.4f} {test_loss_cnn:>14.4f}")
print(f"  {'Trainable Parameters':<30} {nn_params:>15,} {cnn_params:>15,}")
print(f"  {'Parameter Reduction':<30} {'---':>15} {(1 - cnn_params/nn_params)*100:>13.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_nn.history["val_accuracy"],  label="Simple NN", linestyle="--", color="steelblue")
axes[0].plot(history_cnn.history["val_accuracy"], label="CNN",        color="darkorange")
axes[0].set_title("Validation Accuracy Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_nn.history["val_loss"],  label="Simple NN", linestyle="--", color="steelblue")
axes[1].plot(history_cnn.history["val_loss"], label="CNN",        color="darkorange")
axes[1].set_title("Validation Loss Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Simple NN vs CNN - Validation Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("comparison_curves.png", dpi=150)
plt.show()
print("\n  -> Saved: comparison_curves.png")

print("\n" + "=" * 70)
print("  All done! Files saved:")
print("    class_distribution.png")
print("    sample_images.png")
print("    nn_training_curves.png")
print("    nn_confusion_matrix.png")
print("    cnn_training_curves.png")
print("    cnn_confusion_matrix.png")
print("    comparison_curves.png")
print("=" * 70)
