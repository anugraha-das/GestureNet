import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 29

TRAIN_DIR = "dataset/asl_alphabet_train"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...\n")

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("\nClass mapping:")
print(train_data.class_indices)

print("\nTraining images:", train_data.samples)
print("Validation images:", validation_data.samples)


# --------------------------------------------------
# 3. LOAD MOBILENETV2
# --------------------------------------------------

print("\nLoading MobileNetV2...\n")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze MobileNetV2 initially
base_model.trainable = False


# --------------------------------------------------
# 4. ADD ASL CLASSIFICATION LAYERS
# --------------------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

output = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)


# --------------------------------------------------
# 5. COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 6. SHOW MODEL
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# 7. CALLBACKS
# --------------------------------------------------

best_model_path = os.path.join(
    MODEL_DIR,
    "best_mobilenetv2_asl.keras"
)

callbacks = [

    ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]


# --------------------------------------------------
# 8. INITIAL TRAINING
# --------------------------------------------------

print("\n========================================")
print("Starting initial MobileNetV2 training")
print("========================================\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5,
    callbacks=callbacks
)


# --------------------------------------------------
# 9. FINE-TUNING
# --------------------------------------------------

print("\n========================================")
print("Starting MobileNetV2 fine-tuning")
print("========================================\n")

base_model.trainable = True


# Freeze most MobileNetV2 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


# Recompile with smaller learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


fine_tune_history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5,
    callbacks=callbacks
)


# --------------------------------------------------
# 10. SAVE FINAL MODEL
# --------------------------------------------------

final_model_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_asl_final.keras"
)

model.save(final_model_path)


# --------------------------------------------------
# 11. TRAINING COMPLETE
# --------------------------------------------------

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)

print("\nClass mapping:")
print(train_data.class_indices)