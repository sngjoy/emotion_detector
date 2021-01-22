# Emotion Detector
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
Run the detection_emotion_video python file
```
streamlit run src\app.py
```

## Dataset
https://www.kaggle.com/mahmoudima/mma-facial-expression