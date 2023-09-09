import cv2
import numpy as np
from dual_camera import Stereo_Camera

def run_cameras():
    
    window_title = "Dual CSI Cameras"
    camera = Stereo_Camera(calib=True)
    camera.start()

    print("*"*50, camera.getWidth())

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