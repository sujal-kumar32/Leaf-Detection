import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
data_dir = r"C:\Leaf_Dataset"   # If using Colab, upload and change path

# Image settings
img_size = (128, 128)
batch_size = 16

# Create training & validation generators
datagen = ImageDataGenerator(
    rescale=1./255,         # normalize
    validation_split=0.2    # 80% training, 20% testing
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

print("Classes:", train_data.class_indices)
