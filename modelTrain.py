import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Base directory
base_dir = r"C:\Leaf_Dataset"

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

img_size = 128
batch_size = 16

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

print("Classes:", train_gen.class_indices)

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary output
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model (NO VALIDATION)
history = model.fit(
    train_gen,
    epochs=10
)

# Evaluate on test data
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy = {acc*100:.2f}%")

# Save the model
model.save("leaf_model.h5")
print("Model saved as leaf_model.h5")
