from PIL import Image
import os


# List of skin disease classes
data = ['Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions', 'Eczema', 'Melanocytic Nevi', 
        'Melanoma', 'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors', 
        'Tinea Ringworm Candidiasis and other Fungal Infections', 'Warts Molluscum and other Viral Infections']

source_dir = 'data/'  # Path to your original dataset

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    print(f'Processing {class_name}...')
    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            try:
                img_obj = Image.open(img_path)
                img_obj.verify()  # Verify that it is an image
            except Exception as e:
                print(f"Corrupted image: {img_path} - {e}")
