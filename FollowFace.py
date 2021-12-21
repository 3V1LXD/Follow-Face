import cv2
import cvzone
import mediapipe as mp
import dlib
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Track roi points for smoothing
POINTS = []

# Initialize a face cascade using the frontal face haar cascade provided with
# the OpenCV library
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def TrackSmoothing(roi, maxPoints=30):
    top = roi[0]
    bottom = roi[1]
    left = roi[2]
    right = roi[3]
    if len(POINTS) < maxPoints:
        maxPoints = len(POINTS)
    else:
        del POINTS[0]

    POINTS.append([top, bottom, left, right])
    mean = [int((sum(col)/len(col))) for col in zip(*POINTS)]
    return mean


def FollowFace():

    BG_COLOR = (0, 255, 0)  # green

    # Capture video frames
    cap = cv2.VideoCapture(0)

    roi, lastroi = [], []

    # Create opencv named window
    cv2.namedWindow("Follow Face", cv2.WINDOW_NORMAL)

    # Start the window thread for the window we are using
    cv2.startWindowThread()

    # Create the tracker we will use
    tracker = dlib.correlation_tracker()

    # The variable we use to keep track of the fact whether we are
    # currently using the dlib tracker
    trackingFace = 0

    # Variable to track frame count so we can refresh landmark detection
    frames = 0

    with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1) as selfie_segmentation:
        bg_image = None
        while cap.isOpened():
            # Retrieve the latest image from the webcam
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Check if a key was pressed and if it was Q, then destroy all
            # opencv windows and exit the application
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                cv2.destroyAllWindows()
                exit(0)
            elif pressedKey == ord('R'):  # reset landmark detection to refresh img
                roi = []
                trackingFace = 0

            # If we are not tracking a face, then try to detect one
            if not trackingFace:
                # convert the img to gray-based image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # find all faces
                faces = faceCascade.detectMultiScale(
                    gray, minNeighbors=10, minSize=(50, 50), maxSize=(300, 300))

                # get largest face based on the largest
                # area. initialize the required variables to 0
                maxArea = 0
                x = 0
                y = 0
                w = 0
                h = 0

                # Loop over faces and check if the area is the
                # largest convert it to int for dlib tracker.
                for (_x, _y, _w, _h) in faces:
                    if _w*_h > maxArea:
                        x = int(_x)
                        y = int(_y)
                        w = int(_w)
                        h = int(_h)
                        maxArea = w*h

                # If one or more faces are found, initialize the tracker
                # on the largest face
                if maxArea > 0:
                    # Initialize the tracker
                    tracker.start_track(image,
                                        dlib.rectangle(x-10,
                                                       y-20,
                                                       x+w+10,
                                                       y+h+20))

                    # Set the indicator variable when actively tracking region in the image
                    trackingFace = 1
                time.sleep(0.06)

            # Check if the tracker is actively tracking a region in the image
            if trackingFace:

                # Update the tracker and request quality of the tracking update
                trackingQuality = tracker.update(image)

                # determine the updated position of the tracked region
                if trackingQuality >= 8.75:
                    tracked_position = tracker.get_position()

                    # set the ROI as a border around the face
                    t_x = int(tracked_position.left()) - \
                        int(tracked_position.left())/3
                    t_y = int(tracked_position.top()) - \
                        int(tracked_position.top())/3
                    t_w = 320
                    t_h = 320

                    roi = t_y, t_y+t_h, t_x, t_x+t_w

                    if (roi[0] > 0):
                        roi = TrackSmoothing(roi)
                        lastroi = roi

                else:
                    trackingFace = 0

                # reset every 60 frames to refresh face tracking
                frames += 1
                if (frames > 59):
                    frames = 0
                    roi = []
                    trackingFace = 0

            # show the image on the screen
            if image.any():
                if (len(lastroi) > 0):
                    image = image[lastroi[0]:lastroi[1], lastroi[2]:lastroi[3]]

                original = image

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                mask = selfie_segmentation.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw selfie segmentation on the background image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                smoothMask = np.stack(
                    (mask.segmentation_mask,) * 3, axis=-1) > 0.1

                # The background can be customized.
                #   a) Load an image (with the same width and height of the input image) to
                #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
                #   b) Blur the input image by applying image filtering, e.g.,
                #      bg_image = cv2.GaussianBlur(image,(55,55),0)
                # if bg_image is None:
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR

                imgOut = np.where(smoothMask, image, bg_image)

                imgStack = cvzone.stackImages([original, imgOut], 2, 1)
                cv2.imshow("Follow Face", imgStack)
                cv2.resizeWindow("Follow Face", 320*2, 320)

        cap.release()


if __name__ == '__main__':
    FollowFace()
