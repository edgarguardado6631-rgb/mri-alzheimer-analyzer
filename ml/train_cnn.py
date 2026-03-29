import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, r"..\data\Test 2\ADNI")
LABELS_FILE = os.path.join(BASE_DIR, r"..\data\Test_2_2_03_2026.csv")
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 15
LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}

def load_data(data_dir, labels_file, max_samples=100):
    images = []
    labels = [] 
    
    # Load labels
    print(f"Loading labels from {labels_file}...")
    df = pd.read_csv(labels_file)
    # create dictionary mapping Image Data ID -> Group
    id_to_label = dict(zip(df['Image Data ID'], df['Group']))
    
    # Searching for NIfTI files
    print(f"Scanning {data_dir} for .nii files...")
    files = glob.glob(os.path.join(data_dir, "**/*.nii"), recursive=True)
    
    if not files:
        print("No .nii files found directly. Checking .nii.gz")
        files = glob.glob(os.path.join(data_dir, "**/*.nii.gz"), recursive=True)
        
    print(f"Found {len(files)} files. Loading up to {max_samples}...")
    
    for i, file_path in enumerate(files[:max_samples]):
        try:
            # Extract Image Data ID from filename
            # Filename format: ..._I12345.nii or ..._I12345.nii.gz
            # We assume the ID is the last component separated by '_' before extension
            filename = os.path.basename(file_path)
            # handle .nii and .nii.gz
            if filename.endswith('.nii.gz'):
                base_name = filename[:-7]
            elif filename.endswith('.nii'):
                base_name = filename[:-4]
            else:
                continue
                
            image_id = base_name.split('_')[-1]
            
            if image_id not in id_to_label:
                # print(f"Skipping {filename} - Label not found for ID {image_id}")
                continue
                
            group = id_to_label[image_id]
            if group not in LABEL_MAP:
                continue
                
            label = LABEL_MAP[group]

            img = nib.load(file_path)
            data = img.get_fdata()

            # Axis-aware slice selection: find the Superior/Inferior axis from affine
            axcodes = nib.aff2axcodes(img.affine)
            si_axis = next((idx for idx, c in enumerate(axcodes) if c in ['S', 'I']), 2)

            mid_slice_idx = data.shape[si_axis] // 2
            slice_img = (
                data[mid_slice_idx, :, :] if si_axis == 0 else
                data[:, mid_slice_idx, :] if si_axis == 1 else
                data[:, :, mid_slice_idx]
            )

            # Resize, convert to numpy, then normalize
            slice_img = tf.image.resize(slice_img[..., np.newaxis], IMG_SIZE).numpy()
            vmin, vmax = slice_img.min(), slice_img.max()
            slice_img = (slice_img - vmin) / (vmax - vmin + 1e-8)
            
            images.append(slice_img)
            labels.append(label) 
            
            if (i+1) % 50 == 0:
                print(f"Loaded {i+1} images")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    images = np.array(images)
    labels = np.array(labels)
    
    # Compute class weights to handle imbalanced datasets (e.g. fewer AD than CN)
    unique_classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=labels)
    class_weights_dict = dict(zip(unique_classes, weights))
    print(f"Computed class weights: {class_weights_dict}")
            
    return images, labels, class_weights_dict

def create_model():
    # Load VGG16 base model, excluding the top dense layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Freeze all layers except block5 (fine-tune top conv block only)
    UNFREEZE = {'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool'}
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = layer.name in UNFREEZE

    # Create new model on top
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    
    # VGG16 expects 3 channels. We duplicate the grayscale channel to 3 channels using Concatenate.
    x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    
    # Data Augmentation to prevent overfitting on the small dataset
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Pass input through VGG16
    x = base_model(x, training=True) # Let BatchNormalization know training=True if we unfroze stuff
    
    # Add new classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='softmax')(x) # 3 classes: CN, MCI, AD

    model = models.Model(inputs, outputs)
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Starting pipeline...")
    # Increase max_samples to load more data for better accuracy
    X, y, class_weights = load_data(DATA_DIR, LABELS_FILE, max_samples=400)
    
    if len(X) == 0:
        print("No data loaded. Exiting.")
        exit(1)
        
    print(f"Data shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model()
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stopping],
        batch_size=BATCH_SIZE
    )
    
    print("\nEvaluating Model on Test Set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    target_names = ['CN (Cognitively Normal)', 'MCI (Mild Cognitive Impairment)', 'AD (Alzheimer\'s Disease)']
    
    # Generate mapping reverse index to name just to be safe if some classes are missing
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    model_save_path = os.path.join(BASE_DIR, 'alzheimer_cnn_model.h5')
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")
