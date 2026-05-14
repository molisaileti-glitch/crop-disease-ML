# evaluate.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests the trained model on validation images.
# Shows accuracy per disease category and
# sample predictions.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report
from prepare_data import prepare_data
from config import (
    MODEL_SAVE_PATH,
    DISEASE_CLASSES,
    DISEASE_FRIENDLY_NAMES
)


def evaluate():
    """
    Loads the saved model and evaluates it
    showing detailed performance per disease.
    """

    print("=" * 60)
    print("  CROP DISEASE DETECTOR — MODEL EVALUATION")
    print("=" * 60)

    # ── STEP 1: Load validation generator ───────────
    # prepare_data() now returns two generators
    # We only need the validation generator
    print("\n📂 Loading validation data...")
    _, val_generator = prepare_data()

    # ── STEP 2: Load saved model ─────────────────────
    print(f"\n🧠 Loading model from {MODEL_SAVE_PATH}...")
    model = keras.models.load_model(MODEL_SAVE_PATH)
    print("✓ Model loaded successfully")

    # ── STEP 3: Overall accuracy ─────────────────────
    print("\n📊 Evaluating model...")
    val_loss, val_accuracy = model.evaluate(
        val_generator, verbose=1
    )
    print(f"\n✓ Overall Accuracy: {val_accuracy * 100:.2f}%")
    print(f"✓ Overall Loss:     {val_loss:.4f}")

    # ── STEP 4: Per class predictions ────────────────
    print("\n🔍 Getting predictions for each disease...")

    # Reset generator to start from beginning
    val_generator.reset()

    # Get all predictions at once
    predictions = model.predict(val_generator, verbose=1)

    # Get predicted class index for each image
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels from generator
    true_classes = val_generator.classes

    # ── STEP 5: Per disease accuracy ─────────────────
    print("\n📋 Accuracy per disease category:")
    print("-" * 60)

    for class_index, class_name in enumerate(DISEASE_CLASSES):

        # Find all images of this disease
        class_mask = true_classes == class_index
        class_total = np.sum(class_mask)

        if class_total == 0:
            continue

        # Count correct predictions
        class_correct = np.sum(
            predicted_classes[class_mask] == class_index
        )
        class_accuracy = class_correct / class_total * 100

        # Get friendly name
        friendly = DISEASE_FRIENDLY_NAMES.get(
            class_name, class_name
        )

        # Show with emoji based on accuracy
        if class_accuracy >= 90:
            emoji = "🟢"
        elif class_accuracy >= 70:
            emoji = "🟡"
        else:
            emoji = "🔴"

        print(f"  {emoji} {friendly}")
        print(f"      {class_accuracy:.1f}% "
              f"({class_correct}/{class_total} correct)")

    # ── STEP 6: Classification report ────────────────
    print("\n📋 Full Classification Report:")
    print("-" * 60)

    # Short names for display
    short_names = [
        name.split('___')[-1].replace('_', ' ')
        for name in DISEASE_CLASSES
    ]

    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=short_names
    ))

    # ── STEP 7: Sample predictions ───────────────────
    print("\n🖼️  5 Sample Predictions:")
    print("-" * 60)

    # Pick 5 random validation images
    sample_indices = np.random.choice(
        len(true_classes), 5, replace=False
    )

    for idx in sample_indices:
        true_label  = DISEASE_CLASSES[true_classes[idx]]
        pred_label  = DISEASE_CLASSES[predicted_classes[idx]]
        confidence  = predictions[idx][predicted_classes[idx]] * 100
        is_correct  = true_label == pred_label
        symbol      = "✓" if is_correct else "✗"

        print(f"  {symbol} Real:       "
              f"{DISEASE_FRIENDLY_NAMES[true_label]}")
        print(f"    Predicted:  "
              f"{DISEASE_FRIENDLY_NAMES[pred_label]}")
        print(f"    Confidence: {confidence:.1f}%")
        print()

    print("=" * 60)
    print("  EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    evaluate()