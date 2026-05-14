# model.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This file builds our CNN model architecture.
# We use Transfer Learning — meaning we start with
# a model already trained on millions of images
# (MobileNetV2) and fine tune it for our crop
# disease specific task.
#
# WHY TRANSFER LEARNING?
# Training a CNN from scratch needs millions of images
# and weeks of computing time. Transfer learning lets
# us reuse patterns already learned from millions of
# images and just teach the model our specific task.
# This gives us 95%+ accuracy with much less data
# and training time.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from config import IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES, LEARNING_RATE


def build_model():
    """
    Builds and returns the CNN model using
    MobileNetV2 as the base (transfer learning).

    Model structure:
    Input image (224x224x3)
        ↓
    MobileNetV2 base (pre-trained feature extractor)
        ↓
    Global Average Pooling (summarize features)
        ↓
    Dense layer 128 neurons (learn disease patterns)
        ↓
    Dropout (prevent overfitting)
        ↓
    Output layer 15 neurons (one per disease class)
    """

    # ── STEP 1: Load pre-trained MobileNetV2 ────────
    # MobileNetV2 was trained on ImageNet — 1.2 million
    # images across 1000 categories. It already knows
    # how to detect edges, shapes and textures.
    #
    # include_top=False means we remove the last layer
    # (which classified ImageNet categories) and replace
    # it with our own layer for disease classification
    #
    # weights='imagenet' loads the pre-trained weights
    # so we do not start from random values
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS),
        include_top=False,
        weights='imagenet'
    )

    # ── STEP 2: Freeze the base model ───────────────
    # Freeze means we do NOT change the pre-trained
    # weights during training. We only train the new
    # layers we add on top.
    # This prevents us from accidentally destroying
    # the knowledge already learned from ImageNet.
    base_model.trainable = False

    # ── STEP 3: Build our complete model ────────────
    model = keras.Sequential([

        # Input layer — expects 224x224 RGB images
        keras.layers.Input(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS)
        ),

        # MobileNetV2 base — extracts image features
        # Acts like the Conv2D + MaxPooling layers we
        # learned in Phase 5 but much more powerful
        base_model,

        # GlobalAveragePooling — converts feature maps
        # into a single flat vector
        # Same concept as what we learned in NLP lesson
        layers.GlobalAveragePooling2D(),

        # Dense layer — learns disease specific patterns
        # from the features MobileNetV2 extracted
        layers.Dense(128, activation='relu'),

        # Dropout — randomly turns off 30% of neurons
        # during training to prevent overfitting
        # Forces the model to learn robust patterns
        # not just memorize training images
        layers.Dropout(0.3),

        # Output layer — 15 neurons, one per disease
        # Softmax converts outputs to probabilities
        # that all add up to 100%
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    # ── STEP 4: Compile the model ───────────────────
    # optimizer='adam' — adjusts weights during training
    # loss='sparse_categorical_crossentropy' — measures
    #   how wrong predictions are (for multiple classes)
    # metrics=['accuracy'] — track accuracy each epoch
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Run this file to see the model structure
if __name__ == '__main__':
    model = build_model()
    model.summary()
    print(f"\nTotal layers: {len(model.layers)}")
    print(f"Output classes: {NUM_CLASSES}")