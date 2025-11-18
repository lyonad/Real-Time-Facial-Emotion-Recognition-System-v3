# Human Face Emotion Detection

A real-time deep learning computer vision project for detecting human emotions from facial expressions using Convolutional Neural Networks (CNN).

## 🎯 Overview

This project implements a deep learning-based emotion detection system that can classify facial expressions into five emotion categories: **Angry**, **Fear**, **Happy**, **Sad**, and **Surprise**. The system supports both model training and real-time emotion detection using a webcam.

## ✨ Features

- **Multiple CNN Architectures**: Choose from 5 different model architectures optimized for different use cases
- **Real-time Detection**: Live emotion detection using webcam with visual feedback
- **Data Augmentation**: Advanced augmentation pipeline using TensorFlow Dataset API
- **GPU Support**: Optimized for both CPU and GPU training
- **High Accuracy**: Trained on a large dataset with thousands of images per emotion class
- **Easy to Use**: Simple command-line interface for training and inference

## 📋 Requirements

- **Python**: 3.11+ (3.11 or 3.12 recommended for best compatibility)
- **TensorFlow**: 2.20.0+ (<2.21.0)
- **OpenCV**: 4.12.0+ (<5.0.0)
- **NumPy**: 1.26.0+ (<2.2.0 for TensorFlow compatibility)
- **scikit-learn**: 1.6.0+ (<1.7.0)
- **Matplotlib**: 3.10.0+ (<4.0.0)
- **Pillow**: 11.0.0+ (<12.0.0)
- **h5py**: 3.11.0+ (<4.0.0)

All dependencies are specified in `requirements.txt` with version constraints for compatibility.

## 🚀 Installation

### Option 1: Using pip (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd "Human Face Emotions"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

1. Create a new conda environment (Python 3.11 or 3.12 recommended):
```bash
conda create -n emotion_detection python=3.11 -y
conda activate emotion_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Human Face Emotions/
├── Data/                              # Training dataset (ignored by git)
│   ├── Angry/                        # Angry emotion images
│   ├── Fear/                         # Fear emotion images
│   ├── Happy/                        # Happy emotion images
│   ├── Sad/                          # Sad emotion images
│   └── Suprise/                      # Surprise emotion images
├── train_emotion_model.py            # Model training script
├── real_time_emotion_detection.py    # Real-time detection script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
├── README.md                         # Project documentation
│
├── emotion_model.h5                  # Trained model (generated after training)
├── label_encoder.pkl                 # Label encoder (generated after training)
└── training_history.png              # Training history visualization
```

**Note**: The `Data/` directory and `__pycache__/` are ignored by git (see `.gitignore`).

## 🎓 Model Architectures

The project supports 5 different CNN architectures:

1. **v1 - Basic CNN**: Simple convolutional network for quick training
2. **v2 - Enhanced CNN** (Recommended): Improved architecture with batch normalization and dropout
3. **v3 - Residual CNN**: ResNet-inspired architecture with residual connections
4. **v4 - Efficient CNN**: MobileNet-inspired lightweight architecture
5. **v5 - Attention CNN**: SENet-inspired architecture with channel attention mechanisms

Select the architecture in `train_emotion_model.py`:
```python
MODEL_ARCHITECTURE = 'v2'  # Change to 'v1', 'v3', 'v4', or 'v5'
```

## 🏋️ Training

### Basic Training

Train the model with default settings:

```bash
python train_emotion_model.py
```

### Configuration

Edit the following parameters in `train_emotion_model.py`:

```python
IMG_SIZE = 48              # Input image size (48x48)
BATCH_SIZE = 16            # Batch size (16 for CPU, 64-128 for GPU)
EPOCHS = 50                # Number of training epochs
MODEL_ARCHITECTURE = 'v2'  # Model architecture version
```

### Training Output

After training, the script will generate:
- `emotion_model.h5`: Trained model weights
- `label_encoder.pkl`: Label encoder for emotion classes
- `training_history.png`: Training/validation accuracy and loss plots

### GPU Training

For faster training with GPU:

1. Verify GPU availability by checking TensorFlow output when running the training script. The script will automatically detect and report GPU availability.

2. If GPU is available, increase batch size in `train_emotion_model.py`:
```python
BATCH_SIZE = 64  # or 128 for larger GPUs
```

3. For GPU support on Windows, consider using Conda to install TensorFlow with CUDA support.

## 🎥 Real-time Detection

Run real-time emotion detection using your webcam:

```bash
python real_time_emotion_detection.py
```

### Controls

- **Press 'q'**: Quit the application
- **Press 's'**: Save current frame with emotion label

### Features

- Face detection using Haar Cascades
- Real-time emotion classification
- Color-coded emotion labels:
  - 🔴 **Angry**: Red
  - 🟣 **Fear**: Purple
  - 🟢 **Happy**: Green
  - 🔵 **Sad**: Blue
  - 🟠 **Surprise**: Orange

## 📊 Dataset

### Dataset Source

The dataset used in this project is from Kaggle:

**Human Face Emotions Dataset**
- **Source**: [Kaggle - Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)
- **Author**: samithsachidanandan
- **License**: Please check the dataset page for license information

### Dataset Structure

The dataset should be organized in the following structure:

```
Data/
├── Angry/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
├── Fear/
│   └── ...
├── Happy/
│   └── ...
├── Sad/
│   └── ...
└── Suprise/
    └── ...
```

Supported image formats: `.png`, `.jpg`, `.jpeg`

## 🔧 Advanced Usage

### Custom Model Training

You can modify the model architecture by editing the `create_model()` function in `train_emotion_model.py`:

```python
def create_model(architecture='v2', input_shape=(48, 48, 1), num_classes=5):
    # Customize your model here
    ...
```

### Data Augmentation

The training script includes automatic data augmentation using TensorFlow Dataset API:
- Random horizontal flip (50% chance)
- Random brightness adjustment (max delta: 0.1)
- Random contrast adjustment (0.9-1.1 range)

Augmentation is applied automatically during training to improve model generalization.

### Model Evaluation

After training, the script automatically:
- Splits data into train/validation sets (80/20)
- Evaluates on validation set
- Displays accuracy and loss metrics
- Generates training history plots

## 🐛 Troubleshooting

### GPU Not Detected

If GPU is not detected:

1. The training script will automatically check and report GPU availability when you run it.

2. Verify CUDA installation:
```bash
nvcc --version
```

3. For Windows users, TensorFlow from PyPI may be CPU-only. Consider using Conda for GPU support:
```bash
conda install -c conda-forge tensorflow cudatoolkit=11.8 cudnn
```

### Out of Memory Errors

Reduce batch size:
```python
BATCH_SIZE = 8  # or lower
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📈 Performance

- **Input Size**: 48x48 grayscale images
- **Model Size**: ~2-5 MB (depending on architecture)
- **Inference Speed**: ~30-60 FPS on CPU, ~100+ FPS on GPU
- **Accuracy**: Varies by architecture and dataset size

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created for real-time emotion detection using deep learning.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision utilities
- scikit-learn for data preprocessing
- [Human Face Emotions Dataset](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) by samithsachidanandan on Kaggle

---

**Note**: Make sure you have a trained model (`emotion_model.h5`) before running real-time detection. If the model doesn't exist, train it first using `train_emotion_model.py`.

