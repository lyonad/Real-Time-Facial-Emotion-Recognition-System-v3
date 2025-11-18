"""
Training script for real-time emotion detection model
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
try:
    tf.random.set_seed(42)
except:
    pass

# Configuration
IMG_SIZE = 48  # Input image size (48x48 is standard for emotion detection)
BATCH_SIZE = 16  # Reduced for CPU training (increase to 64-128 if GPU available)
EPOCHS = 50
DATA_DIR = "Data"
MODEL_ARCHITECTURE = 'v2'  # Options: 'v1' (Basic), 'v2' (Enhanced - recommended), 'v3' (Residual), 'v4' (Efficient), 'v5' (Attention)

# GPU Optimization (uncomment if using GPU)
# import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Faster training with GPU

# Emotion classes (note: "Suprise" is spelled as in the directory)
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

def load_data(data_dir, img_size=IMG_SIZE):
    """Load images and labels from directory structure"""
    images = []
    labels = []
    
    print("Loading images...")
    for emotion in EMOTIONS:
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: {emotion_path} not found, skipping...")
            continue
            
        emotion_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            emotion_files.extend([f for f in os.listdir(emotion_path) if f.lower().endswith(ext.replace('*', ''))])
        
        print(f"Loading {len(emotion_files)} images for {emotion}...")
        
        for filename in emotion_files:
            try:
                img_path = os.path.join(emotion_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, (img_size, img_size))
                    # Normalize pixel values to [0, 1]
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(emotion)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Reshape images to include channel dimension
    images = images.reshape(images.shape[0], img_size, img_size, 1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print(f"\nLoaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    print(f"Label distribution:")
    for emotion in EMOTIONS:
        count = np.sum(labels == emotion)
        print(f"  {emotion}: {count}")
    
    return images, labels_encoded, label_encoder

def conv_block(x, filters, kernel_size=3, pool_size=2, dropout_rate=0.25):
    """Convolutional block with batch normalization and dropout"""
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

def create_model_v1_basic(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=5):
    """Basic CNN model (original architecture)"""
    inputs = layers.Input(shape=input_shape)
    
    x = conv_block(inputs, 32, dropout_rate=0.25)
    x = conv_block(x, 64, dropout_rate=0.25)
    x = conv_block(x, 128, dropout_rate=0.25)
    x = conv_block(x, 256, dropout_rate=0.25)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def create_model_v2_enhanced(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=5):
    """Enhanced CNN with more filters and better regularization"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution with larger kernel
    x = layers.Conv2D(64, (5, 5), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Convolutional blocks with increasing filters
    x = conv_block(x, 128, dropout_rate=0.3)
    x = conv_block(x, 256, dropout_rate=0.3)
    x = conv_block(x, 512, dropout_rate=0.3)
    
    # Global Average Pooling instead of Flatten (reduces parameters)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with L2 regularization
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def create_model_v3_residual(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=5):
    """Residual CNN with skip connections (ResNet-like)"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial block
    x = layers.Conv2D(64, (7, 7), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual blocks
    def residual_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Match dimensions if needed
        shortcut_channels = keras.backend.int_shape(shortcut)[-1]
        if shortcut_channels != filters:
            shortcut = layers.Conv2D(filters, (1, 1))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def create_model_v4_efficient(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=5):
    """Efficient CNN with depthwise separable convolutions (MobileNet-like)"""
    inputs = layers.Input(shape=input_shape)
    
    def depthwise_separable_conv(x, filters, stride=1):
        # Depthwise convolution
        x = layers.DepthwiseConv2D((3, 3), padding='same', strides=stride)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Pointwise convolution
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    # Initial standard convolution
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Depthwise separable blocks
    x = depthwise_separable_conv(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = depthwise_separable_conv(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = depthwise_separable_conv(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def create_model_v5_attention(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=5):
    """CNN with attention mechanism for better feature focus"""
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional feature extraction
    x = conv_block(inputs, 64, dropout_rate=0.2)
    x = conv_block(x, 128, dropout_rate=0.2)
    x = conv_block(x, 256, dropout_rate=0.3)
    
    # Channel attention (SENet-like)
    def channel_attention(x, ratio=8):
        channels = keras.backend.int_shape(x)[-1]
        # Global average pooling
        gap = layers.GlobalAveragePooling2D()(x)
        gap = layers.Reshape((1, 1, channels))(gap)
        # Dense layers for attention
        fc1 = layers.Dense(channels // ratio, activation='relu')(gap)
        fc2 = layers.Dense(channels, activation='sigmoid')(fc1)
        # Apply attention
        x = layers.Multiply()([x, fc2])
        return x
    
    x = channel_attention(x)
    x = conv_block(x, 512, dropout_rate=0.3)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=5, architecture='v2'):
    """
    Create CNN model for emotion detection
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of emotion classes
        architecture: Model architecture version ('v1', 'v2', 'v3', 'v4', 'v5')
            - v1: Basic CNN (original)
            - v2: Enhanced CNN with better regularization (recommended)
            - v3: Residual CNN with skip connections
            - v4: Efficient CNN with depthwise separable convolutions
            - v5: CNN with attention mechanism
    """
    architectures = {
        'v1': create_model_v1_basic,
        'v2': create_model_v2_enhanced,
        'v3': create_model_v3_residual,
        'v4': create_model_v4_efficient,
        'v5': create_model_v5_attention
    }
    
    if architecture not in architectures:
        print(f"Warning: Architecture '{architecture}' not found. Using 'v2' (Enhanced).")
        architecture = 'v2'
    
    print(f"Creating model with architecture: {architecture.upper()}")
    model = architectures[architecture](input_shape, num_classes)
    return model

def train_model():
    """Main training function"""
    # Load data
    X, y, label_encoder = load_data(DATA_DIR)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Data augmentation function using TensorFlow operations
    def augment_image(image, label):
        """Apply data augmentation to a single image"""
        # Random horizontal flip (50% chance)
        image = tf.image.random_flip_left_right(image)
        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.1)
        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # Random translation (shift) - simulate with random crop and pad
        # Note: For more complex augmentation, consider using tf.keras.layers.RandomTranslation
        # Ensure values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create model
    model = create_model(num_classes=len(EMOTIONS), architecture=MODEL_ARCHITECTURE)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'emotion_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history saved to training_history.png")
    
    # Evaluate model
    print("\nEvaluating model...")
    # Load best weights if available
    if os.path.exists('emotion_model.h5'):
        model.load_weights('emotion_model.h5')
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    print("\nTraining completed!")
    print("Model saved as 'emotion_model.h5'")
    print("Label encoder saved as 'label_encoder.pkl'")

if __name__ == "__main__":
    # Check if GPU is available
    print("=" * 60)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)
    
    try:
        print(f"TensorFlow version: {tf.__version__}")
    except:
        print("TensorFlow loaded")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        print("   Training will use GPU (much faster!)")
        
        # Configure GPU memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"   Warning: {e}")
    else:
        print("⚠️  GPU Not Available - Training will use CPU (slower)")
        print("   To enable GPU, install: pip install tensorflow[and-cuda]")
        print("   See GPU_SETUP.md for detailed instructions")
    
    print("=" * 60)
    print()
    
    train_model()

