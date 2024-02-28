import matplotlib.pyplot as plt
import scienceplots
import cv2
import cv2.aruco as aruco
plt.style.use("science")
plt.ion()

img = cv2.imread("/home/max/labplatform/plots/MA_thesis/materials_methods/aruco_side.png")
aruco_dicts = [cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100),
               cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)]
params = cv2.aruco.DetectorParameters_create()
