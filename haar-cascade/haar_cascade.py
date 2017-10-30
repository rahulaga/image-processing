# usage: haar_cascade.py [-h] -c C [C ...] -i I [I ...]
# 
# Process Haar Cascade
#
# optional arguments:
#  -h, --help    show this help message and exit
#  -c C [C ...]  path to haar cascade xml(s)
#  -i I [I ...]  path to image(s)
#
# author Rahul Agarwal 
# adapted from https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
#

import argparse
import cv2

ap = argparse.ArgumentParser(description="Process Haar Cascade")
ap.add_argument("-c", nargs='+', required=True, help="path to haar cascade xml(s)")
ap.add_argument("-i", nargs='+', required=True, help="path to image(s)")
args = vars(ap.parse_args())

#print(args)

#load images and convert to gray
images = []
images_gray = []
for image_path in args["i"]:
	img = cv2.imread(image_path)
	images.append(img)
	images_gray.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

#load haar xml files
cascades = []
for cascade_path in args["c"]:
	cascades.append(cv2.CascadeClassifier(cascade_path))

#run each cascade on each image
for i in range(0, len(images)):
	for cascade in cascades:
		detect = cascade.detectMultiScale(images_gray[i], scaleFactor=1.3)
		for (x, y, w, h) in detect:
			cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 0, 255), 2)
		
	cv2.imshow("Haar output"+str(i), images[i])
cv2.waitKey(0)
