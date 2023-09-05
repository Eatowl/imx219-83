# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# A simple code snippet
# Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit with two CSI ports (Jetson Nano, Jetson Xavier NX) via OpenCV
# Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# This script will open a window and place the camera stream from each camera in a window
# arranged horizontally.
# The camera streams are each read in their own thread, as when done sequentially there
# is a noticeable lag

import cv2
import threading
import numpy as np

class Camera:

    def __init__(self):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.running = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        
    def start(self, gstreamer_pipeline_string: str):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            
            if self.video_capture != None:
                self.running = True
                self.read_thread = threading.Thread(target=self.updateFrame)
                self.read_thread.start()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)

    def updateFrame(self):
        while self.running:
            grabbed, frame = self.video_capture.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):    
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.running = False
        self.read_thread.join()
        self.read_thread = None

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread != None:
            self.read_thread.join()
        

class Stereo_Camera():

    def __init__(self, calib=False):
        self.left = None
        self.right = None
        self.calib = calib

    def start(self):
        self.left = Camera()
        self.left.start(gstreamer_pipeline(sensor_id=0))

        self.right = Camera()
        self.right.start(gstreamer_pipeline(sensor_id=1))

    def read(self):
        _, left_image = self.left.read()
        _, right_image = self.right.read()
        if self.calib is True:
            left_image, right_image = self.calibCameras(left_image,right_image)

        return left_image, right_image

    def checkVideoCapture(self):
        if self.left.video_capture.isOpened() and\
            self.right.video_capture.isOpened():
            return True
        else: return False

    def loadCalibConfigs(self, extrinsics_path='config/extrinsics.yml',
                               intrinsics_path='config/intrinsics.yml'):
        extrinsics_config = cv2.FileStorage(extrinsics_path, cv2.FILE_STORAGE_READ)
        intrinsics_config = cv2.FileStorage(intrinsics_path, cv2.FILE_STORAGE_READ)

        return extrinsics_config.getNode("R").mat(), extrinsics_config.getNode("T").mat(),\
               extrinsics_config.getNode("R1").mat(), extrinsics_config.getNode("R2").mat(),\
               extrinsics_config.getNode("P1").mat(), extrinsics_config.getNode("P2").mat(),\
               extrinsics_config.getNode("Q").mat(), intrinsics_config.getNode("M1").mat(),\
               intrinsics_config.getNode("D1").mat(), intrinsics_config.getNode("M2").mat(),\
               intrinsics_config.getNode("D2").mat()

    def calibCameras(self, left_image, right_image):
        R, T, R1, R2, P1, P2, Q, M1, D1, M2, D2 = self.loadCalibConfigs('config/extrinsics.yml',
                                                                        'config/intrinsics.yml')
        if self.left != None:
            height, width, channel = left_image.shape

            leftMapX, leftMapY = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (width, height), cv2.CV_32FC1)
            left_rectified = cv2.remap(left_image, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            rightMapX, rightMapY = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (width, height), cv2.CV_32FC1)
            right_rectified = cv2.remap(right_image, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            return left_rectified, right_rectified
        else:
            return False

    def stop(self):
        self.left.stop()
        self.left.release()
        self.right.stop()
        self.right.release()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return "nvarguscamerasrc sensor-id=%d ! " \
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! " \
        "nvvidconv flip-method=%d ! " \
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! " \
        "videoconvert ! " \
        "video/x-raw, format=(string)BGR ! appsink" \
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    

def run_cameras():
    
    window_title = "Dual CSI Cameras"
    camera = Stereo_Camera(calib=True)
    camera.start()

    if camera.checkVideoCapture():
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                left_image, right_image = camera.read()
                camera_images = np.hstack((left_image, right_image))

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, camera_images)
                else:
                    break

                keyCode = cv2.waitKey(30) & 0xFF
                if keyCode == 27:
                    break
        finally:
            camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")
        camera.stop()


if __name__ == "__main__":
    run_cameras()
