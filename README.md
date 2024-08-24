

# Image Classification with Convolutional Neural Networks (CNN)


## Project Overview
This project involves building and training a Convolutional Neural Network (CNN) model to classify images into one of 12 classes. The model is implemented using Keras, with data augmentation and preprocessing handled by the `ImageDataGenerator` class.

## Features
- **CNN Model**: The model consists of multiple convolutional and pooling layers, followed by dense layers.
- **Data Augmentation**: The model uses `ImageDataGenerator` for real-time data augmentation during training.
- **Customizable Parameters**: Users can adjust image size, batch size, and number of epochs.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Keras, TensorFlow, NumPy, PIL (Python Imaging Library)
- **Frameworks**: Keras with TensorFlow backend

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- PIL (Python Imaging Library)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/image-classification-cnn.git
   cd image-classification-cnn
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install tensorflow keras numpy pillow
   ```

## Dataset
The project expects a dataset organized into subdirectories, where each subdirectory contains images belonging to a specific class. For example:

```
data/
│
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
├── class2/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── ...
```
- The dataset should be placed in the `data/` directory under the `base_path`.

## Model Architecture
The CNN model architecture consists of:
- **Convolutional Layers**: Extracts features from the input images using filters.
- **Pooling Layers**: Reduces the spatial dimensions of the feature maps.
- **Fully Connected Layers**: Performs the classification based on the extracted features.

Here's a summary of the layers:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(12, activation='softmax')
])
```

## Training
To train the model, execute the following steps:

1. **Set the base path to your dataset** in the script:
   ```python
   base_path = "C:/Users/Vignesh/A/"
   ```

2. **Run the training script**:
   ```bash
   python train_model.py
   ```

### Hyperparameters
- **Image Size**: 150x150 pixels
- **Batch Size**: 20
- **Epochs**: 12

## Evaluation
After training, the model's performance is evaluated on a validation set. The accuracy and loss metrics are tracked throughout the training process.

## Usage
After training, the model is saved as `train_model.h5`. You can load and use this model for predictions on new images.

### Load and Predict
```python
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('train_model.h5')
img = image.load_img('path_to_image.jpg', target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(prediction)
```

## Project Structure
```
image-classification-cnn/
│
├── data/                # Dataset directory
├── train_model.py       # Training script
├── train_model.h5       # Trained model (saved after training)
├── requirements.txt     # Python dependencies
└── README.md            # This README file
```

## Contributing
Contributions are welcome! Please create a pull request with your changes or submit an issue if you find a bug.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the developers of Keras and TensorFlow for providing powerful tools for deep learning.
- Special thanks to the open-source community for datasets and resources.

