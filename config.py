# config.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# All settings for our project in one place.
# Change values here and they update everywhere.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os

# ── PATHS ───────────────────────────────────────────
# Base directory where this file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to PlantVillage dataset
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'PlantVillage')

# Where to save the trained model
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'crop_disease_model.h5')

# Where to save label names so Django can read them later
LABELS_SAVE_PATH = os.path.join(BASE_DIR, 'class_labels.json')

# ── IMAGE SETTINGS ──────────────────────────────────
# Resize every image to 224x224 pixels before training
# This is standard for most CNN models
IMAGE_SIZE = (224, 224)

# 3 channels = RGB color image
IMAGE_CHANNELS = 3

# ── TRAINING SETTINGS ───────────────────────────────
# Process 32 images at a time
BATCH_SIZE = 32

# Model sees all training data 20 times
EPOCHS = 20

# 20% of data for validation, 80% for training
VALIDATION_SPLIT = 0.2

# ── MODEL SETTINGS ──────────────────────────────────
# Standard learning rate for Adam optimizer
LEARNING_RATE = 0.001

# ── DISEASE CATEGORIES ──────────────────────────────
# Exact folder names from our dataset
# Model learns to classify into these 15 categories
DISEASE_CLASSES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]

# Total number of categories
NUM_CLASSES = len(DISEASE_CLASSES)

# ── FRIENDLY NAMES ──────────────────────────────────
# Human readable names for each disease
# These are what the farmer sees in the app
DISEASE_FRIENDLY_NAMES = {
    'Pepper__bell___Bacterial_spot':
        'Pepper — Bacterial Spot',
    'Pepper__bell___healthy':
        'Pepper — Healthy',
    'Potato___Early_blight':
        'Potato — Early Blight',
    'Potato___Late_blight':
        'Potato — Late Blight',
    'Potato___healthy':
        'Potato — Healthy',
    'Tomato_Bacterial_spot':
        'Tomato — Bacterial Spot',
    'Tomato_Early_blight':
        'Tomato — Early Blight',
    'Tomato_Late_blight':
        'Tomato — Late Blight',
    'Tomato_Leaf_Mold':
        'Tomato — Leaf Mold',
    'Tomato_Septoria_leaf_spot':
        'Tomato — Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite':
        'Tomato — Spider Mites',
    'Tomato__Target_Spot':
        'Tomato — Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus':
        'Tomato — Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus':
        'Tomato — Mosaic Virus',
    'Tomato_healthy':
        'Tomato — Healthy',
}

# ── TREATMENT RECOMMENDATIONS ───────────────────────
# What to tell the farmer after diagnosis
# This appears in the Flutter app after prediction
DISEASE_TREATMENTS = {
    'Pepper__bell___Bacterial_spot':
        'Remove infected leaves immediately. Apply copper-based bactericide spray. Avoid overhead watering. Rotate crops next season.',
    'Pepper__bell___healthy':
        'Your pepper plant looks healthy! Continue regular watering and monitoring.',
    'Potato___Early_blight':
        'Apply fungicide containing chlorothalonil or mancozeb. Remove infected leaves. Ensure good air circulation between plants.',
    'Potato___Late_blight':
        'URGENT: This disease spreads very fast. Remove and destroy all infected plants immediately. Apply metalaxyl fungicide. Do not compost infected material.',
    'Potato___healthy':
        'Your potato plant looks healthy! Monitor regularly for any signs of disease.',
    'Tomato_Bacterial_spot':
        'Apply copper-based bactericide. Remove infected leaves. Avoid working with plants when wet. Space plants for good airflow.',
    'Tomato_Early_blight':
        'Apply mancozeb or chlorothalonil fungicide. Remove lower infected leaves. Mulch around base of plant to prevent soil splash.',
    'Tomato_Late_blight':
        'URGENT: Apply metalaxyl fungicide immediately. Remove infected plant parts. This disease can destroy entire crop within days.',
    'Tomato_Leaf_Mold':
        'Improve air circulation. Reduce humidity. Apply fungicide with chlorothalonil. Remove infected leaves.',
    'Tomato_Septoria_leaf_spot':
        'Remove infected leaves. Apply mancozeb fungicide. Avoid overhead watering. Rotate crops next season.',
    'Tomato_Spider_mites_Two_spotted_spider_mite':
        'Apply miticide or insecticidal soap. Spray undersides of leaves. Increase humidity around plants. Remove heavily infected leaves.',
    'Tomato__Target_Spot':
        'Apply fungicide with azoxystrobin. Remove infected leaves. Improve air circulation. Avoid overhead irrigation.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus':
        'No cure available. Remove and destroy infected plants immediately to prevent spread. Control whitefly population which spreads this virus. Use reflective mulch.',
    'Tomato__Tomato_mosaic_virus':
        'No cure available. Remove infected plants. Wash hands and tools after handling. Control aphids which spread this virus. Plant resistant varieties next season.',
    'Tomato_healthy':
        'Your tomato plant looks healthy! Continue regular watering, fertilizing and monitoring.',
}

# ── SEVERITY LEVELS ─────────────────────────────────
# How serious each disease is
# Shown in the Flutter app to help farmer prioritize
DISEASE_SEVERITY = {
    'Pepper__bell___Bacterial_spot': 'Medium',
    'Pepper__bell___healthy':        'None',
    'Potato___Early_blight':         'Medium',
    'Potato___Late_blight':          'High',
    'Potato___healthy':              'None',
    'Tomato_Bacterial_spot':         'Medium',
    'Tomato_Early_blight':           'Medium',
    'Tomato_Late_blight':            'High',
    'Tomato_Leaf_Mold':              'Low',
    'Tomato_Septoria_leaf_spot':     'Medium',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Medium',
    'Tomato__Target_Spot':           'Medium',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'High',
    'Tomato__Tomato_mosaic_virus':   'High',
    'Tomato_healthy':                'None',
}