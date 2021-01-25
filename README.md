# Emotions Detector
### Introduction
This project aims to detect and classify emotions of a person's face into one of the 5 classes of emotions - angry, happy, neutral, sad and surprise - using deep convolutional neural networks.

Tensorflow was used to build and train the CNN model. OpenCV was used to work with video streams and video files. Streamlit was used to build the front end web application.

### Basic Usage
#### Step 1:
Clone the repository 
```
git clone https://github.com/sngjoy/emotion_detector.git
cd emotion_detector
```
#### Step 2:
Install required packages

```
pip install -r requirements.txt
```

#### Step 3: 
Run the app!
```
streamlit run src/app.py
```

If the app does not work, you could run it as a python module.
```
python src/detect_emotion_video.py
```

## About the Model
##### Model Architecture
1. Object localisation:
A pretrained face detector from OpenCV is used.
It is based on Single Shot Detector (SSD) framework with a ResNet base network.
The model can be found in the `face_detector` folder.
    - `deploy.prototxt`: contains ResNet model architecture
    - `res10_300x300_ssd_iter_140000.caffemodel`: contains weights of the model


2. Object Classification
The CNN model consists of the following layers:
    - Convolution (64 Filters)
        - Conv2D
        - Batch Norm 
        - Conv2D
        - Batch Norm
        - Max Pool
        - Dropout
    - Convolution (128 Filters)
    - Convolution (256 Filters)
    - FC Batch Norm Dropout
    - Softmax Output 
    
The model can be found in the `model` folder.

##### Model Parameters
Loss function: Categorical Cross Entropy
- Optimizer: Adam
- Learning Rate: 0.0001
- Batch Size: 512
- Train-Val-Test Split:
    - Training Images: 84,878
    - Validation Images: 16,041
    - Testing Images: 16,010

##### Model Performance
The current model for emotions classification `emo.h5` has a test accuracy of 66%.  

## Dataset
Dataset used to train the model can be downloaded from https://www.kaggle.com/mahmoudima/mma-facial-expression

Data augmentation is used on the train set with the following parameters:
- Rotation Range: 15 degrees
- Width Shift: 0.15
- Height Shift: 0.15
- Shear range: 0.15
- Zoom range: 0.15
- Horizontal Flip: True

To retrain model, simple download the dataset and place it in a `data` folder in the main directory and edit the `dataloader.py` and `model.py` files in `src` folder accordingly.

## Future Developments
- Improving on model accuracy
- Adding more emotion classes