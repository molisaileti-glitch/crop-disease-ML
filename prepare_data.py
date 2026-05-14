# prepare_data.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Instead of loading all images into RAM at once
# we use ImageDataGenerator which loads images
# in small batches directly from disk during training.
# This solves the memory problem completely.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import json
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    DATASET_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    DISEASE_CLASSES,
    LABELS_SAVE_PATH,
    BASE_DIR
)

# Paths for train and validation folders
# We will split the dataset into these two folders
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR   = os.path.join(BASE_DIR, 'data', 'val')


def split_dataset():
    """
    Splits the dataset into train and validation
    folders. 80% goes to train, 20% goes to val.
    Only runs once — skips if already split.
    """

    # Check if already split — skip if yes
    if os.path.exists(TRAIN_DIR):
        print("✓ Dataset already split — skipping")
        return

    print("Splitting dataset into train/val folders...")
    print("This runs once only.")
    print("-" * 50)

    for class_name in DISEASE_CLASSES:

        # Source folder in original dataset
        source_folder = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(source_folder):
            print(f"⚠️  Skipping {class_name} — not found")
            continue

        # Get all image files in this folder
        all_images = [
            f for f in os.listdir(source_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Shuffle images randomly
        np.random.seed(42)
        np.random.shuffle(all_images)

        # Calculate split point
        # 80% for training, 20% for validation
        split_point = int(len(all_images) * (1 - VALIDATION_SPLIT))
        train_images = all_images[:split_point]
        val_images   = all_images[split_point:]

        # Create destination folders
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir   = os.path.join(VAL_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy training images
        for img in train_images:
            src = os.path.join(source_folder, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)

        # Copy validation images
        for img in val_images:
            src = os.path.join(source_folder, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)

        print(f"  ✓ {class_name}:")
        print(f"      Train: {len(train_images)} | Val: {len(val_images)}")

    print("-" * 50)
    print("✓ Dataset split complete!")


def save_class_labels():
    """
    Saves disease class names to JSON file.
    Django reads this to know what each
    prediction number means.
    """
    labels_dict = {
        str(i): class_name
        for i, class_name in enumerate(DISEASE_CLASSES)
    }
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    print(f"✓ Class labels saved to {LABELS_SAVE_PATH}")


def create_generators():
    """
    Creates ImageDataGenerators that load images
    in small batches from disk during training.
    This uses almost zero RAM regardless of
    how large the dataset is.

    Also applies data augmentation to training images —
    this means randomly flipping, rotating and zooming
    images to create variety and prevent overfitting.
    """

    # ── Training generator ───────────────────────────
    # Applies augmentation — creates variety in training
    # data by randomly transforming images
    train_datagen = ImageDataGenerator(
        # Scale pixels from 0-255 to 0-1
        rescale=1./255,

        # Randomly flip image horizontally
        # A leaf flipped sideways is still the same disease
        horizontal_flip=True,

        # Randomly rotate image up to 20 degrees
        rotation_range=20,

        # Randomly zoom in or out up to 20%
        zoom_range=0.2,

        # Randomly shift image horizontally up to 20%
        width_shift_range=0.2,

        # Randomly shift image vertically up to 20%
        height_shift_range=0.2,
    )

    # ── Validation generator ─────────────────────────
    # NO augmentation for validation — only scale pixels
    # We want to test on real unmodified images
    val_datagen = ImageDataGenerator(rescale=1./255)

    # ── Load images from folders ─────────────────────
    # flow_from_directory automatically reads class names
    # from folder names and assigns labels
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        # Resize all images to 224x224
        target_size=IMAGE_SIZE,
        # Load 32 images at a time
        batch_size=BATCH_SIZE,
        # sparse = labels are integers (0, 1, 2...)
        class_mode='sparse',
        # Shuffle training data each epoch
        shuffle=True,
        # For reproducibility
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        # Do not shuffle validation data
        shuffle=False
    )

    print(f"\n✓ Training batches:   {len(train_generator)}")
    print(f"✓ Validation batches: {len(val_generator)}")
    print(f"✓ Class indices: {train_generator.class_indices}")

    return train_generator, val_generator


def prepare_data():
    """
    Main function:
    1. Splits dataset into train/val folders
    2. Creates batch generators
    3. Saves class labels
    """

    # Step 1 — Split dataset
    split_dataset()

    # Step 2 — Save class labels for Django
    save_class_labels()

    # Step 3 — Create generators
    train_generator, val_generator = create_generators()

    return train_generator, val_generator


# Test data preparation
if __name__ == '__main__':
    train_gen, val_gen = prepare_data()
    print(f"\nTraining samples:   {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Number of classes:  {train_gen.num_classes}")