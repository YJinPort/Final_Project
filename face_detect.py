# MIT License
# Copyright (c) 2019-2022 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV

import cv2
import threading
import numpy as np

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1920x1080 @ 30fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
# Notice that we drop frames if we fall outside the processing time in the appsink element

class CSI_Camera:
	def __init__(self):
		self.video_capture = None
		self.frame = None
		self.grabbed = False
		self.read_thread = None
		self.read_lock = threading.Lock()
		self.running = False
	
	def open(self, gstreamer_pipeline_string):
		try:
			self.video_capture = cv2.VideoCapture(
				gstreamer_pipeline_string, cv2.CAP_GSTREAMER
			)
			self.grabbed, self.frame = self.video_capture.read()
		
		except RuntimeError:
			self.video_capture = None
			print("Unable to open camera")
			print("Pipeline: " + gstreamer_pipeline_string)

	def start(self):
		if self.running:
			print("Video capturing is already running")
			return None

		if self.video_capture != None:
			self.running = True
			self.read_thread = threading.Thread(target=self.updateCamera)
			self.read_thread.start()
		return self

	def stop(self):
		self.running = False
		self.read_thread.join()
		self.read_thread = None

	def updateCamera(self):
		while self.running:
			try:
				grabbed, frame = self.video_capture.read()
				with self.read_lock:
					self.grabbed = grabbed
					self.frame = frame
			except RuntimeError:
				print("Could not read image from camera")

	def read(self):
		with self.read_lock:
			frame = self.frame.copy()
			grabbed = self.grabbed
		return grabbed, frame

	def release(self):
		if self.video_capture != None:
			self.video_capture.release()
			self.video_capture = None

		if self.read_thread != None:
			self.read_thread.join()

def gstreamer_pipeline(
	sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
			sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def face_detect():
    window_title = "Face Detect"

    left_camera = CSI_Camera()
    left_camera.open(
		gstreamer_pipeline(
			sensor_id=0,
			capture_width=1280,
			capture_height=720,
			flip_method=0,
			display_width=640,
			display_height=360,
		)
	)
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
		gstreamer_pipeline(
			sensor_id=1,
			capture_width=1280,
			capture_height=720,
			flip_method=0,
			display_width=640,
			display_height=360,
		)
    )
    right_camera.start()

    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"
    )

#video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
#ret, frame = video_capture.read()
                ret1, left_image = left_camera.read()
                ret2, right_image = right_camera.read()
				
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
                faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
                faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

#camera_images = np.hstack((left_image, right_image))

                for (x1, y1, w1, h1) in faces1:
                    cv2.rectangle(left_image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
                    roi_gray1 = gray1[y1 : y1 + h1, x1 : x1 + w1]
                    roi_color1 = left_image[y1 : y1 + h1, x1 : x1 + w1]
                    eyes1 = eye_cascade.detectMultiScale(roi_gray1)
                    for (ex1, ey1, ew1, eh1) in eyes1:
                        cv2.rectangle(
                            roi_color1, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (0, 255, 0), 2
                        )

                for (x2, y2, w2, h2) in faces2:
                    cv2.rectangle(right_image, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
                    roi_gray2 = gray2[y2 : y2 + h2, x2 : x2 + w2]
                    roi_color2 = right_image[y2 : y2 + h2, x2 : x2 + w2]
                    eyes2 = eye_cascade.detectMultiScale(roi_gray2)
                    for (ex2, ey2, ew2, eh2) in eyes2:
                        cv2.rectangle(
                            roi_color2, (ex2, ey2), (ex2 + ew2, ey2 + eh2), (0, 255, 0), 2
                        )
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    camera_images = np.hstack((left_image, right_image))
                    cv2.imshow(window_title, camera_images)
                else:
                    break
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
#video_capture.release()
#left_camera.stop()
            left_camera.release()
#right_camera.stop()
            right_camera.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
#left_camera.stop()
#left_camera.release()
#right_camera.stop()
#right_camera.release()


if __name__ == "__main__":
    face_detect()
