from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import os
import cv2
import time

detections = None 
def detect_and_predict_emotions(frame, faceNet, emotionsNet, threshold):
    # grab the dimensions of the frame and then construct a blob
    # from it
    global detections 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our emotions network
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 24x24, and preprocess it
            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (48, 48)) # the input size for our model
                face = img_to_array(face)
                # face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)            
                # add the face and bounding boxes to their respective lists
                locs.append((startX, startY, endX, endY))
                preds.append(emotionsNet.predict(face)[0].tolist())
            except:
                continue
    return (locs, preds)

def start_video(faceNet, emotionsNet):
    labels=["angry", "happy", "neutral", "sad", "surprise"]
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(0).start()
    time.sleep(2.0)
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        original_frame = frame.copy()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        
        # detect faces in the frame and determine their emotions
        (locs, preds) = detect_and_predict_emotions(frame, faceNet, emotionsNet, 0.5)

        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            # include the probability in the label
            argmaxIdx = np.argmax(pred)
            # some engineering here to increase threshold for neutral 
            # and lower the thresholds for angry and sad 
            # TODO: Train a more accurate model...
            if pred[0] > 0.10 and str(labels[argmaxIdx]) == 'neutral':	# angry
                label = str(labels[0])
                prob  = pred[argmaxIdx]
            elif pred[3] > 0.15 and str(labels[argmaxIdx]) == 'neutral':	# sad
                label = str(labels[3])
                prob = pred[argmaxIdx]
            else: 
                label = str(labels[argmaxIdx])
                prob = pred[argmaxIdx]
            print('')
            # display the label and bounding box rectangle on the output frame
            if label == "happy":
                cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,200,50), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(0,200,50), 2)
            elif label == "neutral":
                cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(255,255,255), 2)
            elif label == "sad":
                cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(200,50,0), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(200,50,0), 2)
            elif label == "angry":
                cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0, 255), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(0,0, 255), 2)
            elif label == "surprise":
                cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,0), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(255,255,0), 2)
        frame = cv2.resize(original_frame,(860,490))
        cv2.imshow("Emotion Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    # SETTINGS
    EMOTIONS_MODEL_PATH=os.getcwd()+"/model/emo.h5"
    FACE_MODEL_PATH=os.getcwd()+"/face_detector" 
    THRESHOLD = 0.5

    # load our serialized face detector model
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
    weightsPath = os.path.sep.join([FACE_MODEL_PATH,"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the emotions detector model
    print("[INFO] loading emotion detector model...")
    emotionsNet = load_model(EMOTIONS_MODEL_PATH)

    start_video(faceNet, emotionsNet)