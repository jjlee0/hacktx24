import tensorflow as tf
from tensorflow.keras import layers, models, Input
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# List of skin disease classes
data = ['Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions', 'Eczema', 'Melanocytic Nevi', 
        'Melanoma', 'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors', 
        'Tinea Ringworm Candidiasis and other Fungal Infections', 'Warts Molluscum and other Viral Infections']

steps_per_epoch = 1000  # Adjust based on dataset size and batch size


# Paths to the dataset and split folders
source_dir = 'data/'  # Main dataset directory containing class folders
output_dir = 'data_split/'  # Directory for train/test split
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Define function to check if an image is valid
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Check if it's an actual image
        return True
    except (IOError, SyntaxError):
        return False

# Clear existing train/test split folders if they exist
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create main train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each class and split images
test_size = 0.2  # 20% for testing

print('Splitting data into train and test sets...')

for class_name in data:
    class_path = os.path.join(source_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    print(f'Processing {class_name}...')
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)
    
    if os.path.isdir(class_path):
        # Filter to include only valid image files
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and is_valid_image(os.path.join(class_path, f))]
        print(f'Found {len(images)} valid images in {class_name}')
        
        # Split images into train and test sets
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        
        # Copy images to respective train/test directories with error handling
        for img in train_images:
            try:
                shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
            except Exception as e:
                print(f'Error copying {img} to {train_class_dir}: {e}')
        
        for img in test_images:
            try:
                shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))
            except Exception as e:
                print(f'Error copying {img} to {test_class_dir}: {e}')

        # Debug print to confirm images copied
        print(f'Copied {len(train_images)} to {train_class_dir}')
        print(f'Copied {len(test_images)} to {test_class_dir}')
    else:
        print(f'No images found in {class_path}')

# Custom generator function for loading images with error handling
def data_generator(file_paths, labels, image_size=(128, 128)):
    for file_path, label in zip(file_paths, labels):
        try:
            img = Image.open(file_path).convert('RGB')  # Open image and convert to RGB
            img = img.resize(image_size)               # Resize to desired size
            img_array = np.array(img) / 255.0          # Normalize the image
            yield img_array, label                     # Yield the image and label
        except Exception as e:
            print(f"Skipping corrupted image: {file_path} ({e})")
            continue

# Prepare file paths and labels for train and test sets
train_files = []
train_labels = []
test_files = []
test_labels = []

for i, class_name in enumerate(data):
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    train_files.extend([os.path.join(train_class_dir, img) for img in os.listdir(train_class_dir)])
    train_labels.extend([i] * len(os.listdir(train_class_dir)))
    test_files.extend([os.path.join(test_class_dir, img) for img in os.listdir(test_class_dir)])
    test_labels.extend([i] * len(os.listdir(test_class_dir)))

# Create TensorFlow datasets from the generator
train_ds = tf.data.Dataset.from_generator(
    lambda: data_generator(train_files, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).repeat().batch(32).prefetch(tf.data.AUTOTUNE)  # Added .repeat() to ensure each epoch has enough steps

test_ds = tf.data.Dataset.from_generator(
    lambda: data_generator(test_files, test_labels),
    output_signature=(
        tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).batch(32).prefetch(tf.data.AUTOTUNE)


# Define the CNN model
model = models.Sequential([
    Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(data), activation='softmax')  # Use len(data) for number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=10
)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test accuracy: {test_accuracy}')

# Save the model
model.save('model_current.keras')

