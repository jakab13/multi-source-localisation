import cv2
from headpose.detect import PoseEstimator
from matplotlib import pyplot as plt

est = PoseEstimator()  #load the model
# take an image using the webcam (alternatively, you could load an image)
cam = cv2.VideoCapture(0)
for i in range(cv2.CAP_PROP_FRAME_COUNT):
    cam.grab()
ret, image = cam.retrieve()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

est.detect_landmarks(image, plot=True)  # plot the result of landmark detection
roll, pitch, yaw = est.pose_from_image(image)  # estimate the head pose

cam.release()
