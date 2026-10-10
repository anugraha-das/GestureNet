
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

MODEL_PATH = "model/best_mobilenetv2_asl.keras"
TEST_DIR = "dataset/asl_alphabet_test"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# --------------------------------------------------
# 3. FIND TEST IMAGES
# --------------------------------------------------

image_files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    and f.lower().endswith("_test.jpg")
]

if not image_files:
    raise FileNotFoundError(
        f"No *_test.jpg images found in {TEST_DIR}"
    )

# --------------------------------------------------
# 4. PREDICT EACH IMAGE
# --------------------------------------------------

correct = 0
total = 0

print("\nIndividual predictions:\n")

for filename in sorted(image_files):
    # Example: A_test.jpg -> A
    true_label = filename[:-9]

    if true_label not in CLASS_NAMES:
        print(f"Skipping unknown label: {filename}")
        continue

    image_path = os.path.join(TEST_DIR, filename)

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Match MobileNetV2 preprocessing used during training
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    is_correct = predicted_label == true_label

    if is_correct:
        correct += 1

    total += 1

    result = "CORRECT" if is_correct else "WRONG"

    print(
        f"{filename:20s} "
        f"Actual: {true_label:8s} "
        f"Predicted: {predicted_label:8s} "
        f"Confidence: {confidence:6.2f}% "
        f"{result}"
    )

# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

if total > 0:
    accuracy = (correct / total) * 100

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Test accuracy: {accuracy:.2f}%")
else:
    print("No valid test images were evaluated.")