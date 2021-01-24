FROM python:3.7

COPY src src/
COPY face_detector face_detector/
COPY model model/
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8501
CMD  ["streamlit", "run", "src/app.py"]

# TODO: figure out how to import cv2