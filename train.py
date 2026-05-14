# train.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main training file — updated to work with
# ImageDataGenerator (batch loading from disk).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from prepare_data import prepare_data
from model import build_model
from config import EPOCHS, MODEL_SAVE_PATH


def create_callbacks():
    """
    Callbacks run automatically after each epoch.
    They save the best model, stop early if needed,
    and reduce learning rate when stuck.
    """

    # Save model only when validation accuracy improves
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Stop training if no improvement for 5 epochs
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )

    return [checkpoint, early_stopping, reduce_lr]


def plot_training_history(history):
    """
    Draws accuracy and loss graphs
    and saves them as PNG files.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy graph
    ax1.plot(history.history['accuracy'],
             label='Training', color='blue')
    ax1.plot(history.history['val_accuracy'],
             label='Validation', color='orange')
    ax1.set_title('Accuracy Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss graph
    ax2.plot(history.history['loss'],
             label='Training', color='blue')
    ax2.plot(history.history['val_loss'],
             label='Validation', color='orange')
    ax2.set_title('Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("✓ Training graph saved to training_history.png")


def train():
    """
    Complete training pipeline.
    """

    print("=" * 60)
    print("  CROP DISEASE DETECTOR — MODEL TRAINING")
    print("=" * 60)

    # Step 1 — Prepare data generators
    print("\n📂 STEP 1: Preparing data generators...")
    train_generator, val_generator = prepare_data()

    # Step 2 — Build model
    print("\n🧠 STEP 2: Building CNN model...")
    model = build_model()
    model.summary()

    # Step 3 — Create callbacks
    print("\n⚙️  STEP 3: Setting up callbacks...")
    callbacks = create_callbacks()

    # Step 4 — Train
    print(f"\n🚀 STEP 4: Training for up to {EPOCHS} epochs...")
    print(f"   Training samples:   {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print("-" * 60)

    history = model.fit(
        # Training data generator
        train_generator,
        # Maximum epochs
        epochs=EPOCHS,
        # Validation data generator
        validation_data=val_generator,
        # Callbacks
        callbacks=callbacks,
        # Show progress
        verbose=1
    )

    # Step 5 — Final evaluation
    print("\n📊 STEP 5: Final evaluation...")
    val_loss, val_accuracy = model.evaluate(
        val_generator, verbose=0
    )
    print(f"   Final Accuracy: {val_accuracy * 100:.2f}%")
    print(f"   Final Loss:     {val_loss:.4f}")

    # Step 6 — Save graphs
    print("\n📈 STEP 6: Saving training graphs...")
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Model saved: {MODEL_SAVE_PATH}")
    print("=" * 60)

    return model, history


if __name__ == '__main__':
    train()