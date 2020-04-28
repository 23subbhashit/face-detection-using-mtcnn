# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:38:37 2020

@author: SUNIL
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
def draw_image_with_boxes(img, result_list):
	plt.imshow(img)
	ax = plt.gca()
	for result in result_list:
		x, y, width, height = result['box']
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		ax.add_patch(rect)
		for key, value in result['keypoints'].items():
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	plt.show()
cap = cv2.VideoCapture(0)

while 1:
    _, img = cap.read()
    img = cv2.flip(img,+1)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    draw_image_with_boxes(img, faces)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
