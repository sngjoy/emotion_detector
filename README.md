# Emotions Detector
### Introduction
This project aims to detect and classify emotions of a person's face into one of the 5 emotion classes - Happiness, Neutral, Sadness, Angry and Surprised - using deep convolutional neural networks.

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
Option 1: Run it as a local web app
```
streamlit run src\app.py
```

Option 2: Run it as a python module
```
python src\detect_emotion_video.py
```

## Model
The current model `emo.h5` has an accuracy of 66%.  

## Dataset
Dataset is obtained from https://www.kaggle.com/mahmoudima/mma-facial-expression
To retrain model, edit `model.py` in `src` folder and place the downloaded dataset in a `data` folder in the main directory.

## Future Developments
- Improving on model accuracy
- Adding more emotion classes