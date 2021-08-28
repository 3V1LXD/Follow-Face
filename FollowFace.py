import cv2
import dlib
import time
POINTS = []


def followFaceSmoothing(roi, maxPoints=30):
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


# Initialize a face cascade using the frontal face haar cascade provided with
# the OpenCV library
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detectAndTrackLargestFace():
    # Capture video frames
    capture = cv2.VideoCapture(0)
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

    try:
        while True:
            # Retrieve the latest image from the webcam
            rc, img = capture.read()

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
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                # largest convert it to int fordlib tracker.
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
                    tracker.start_track(img,
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
                trackingQuality = tracker.update(img)

                # determine the updated position of the tracked region and crop
                if trackingQuality >= 8.75:
                    tracked_position = tracker.get_position()

                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())
                    roi = t_y, t_y+t_h, t_x, t_x+t_w
                    if (roi[0] > 0):
                        roi = followFaceSmoothing(roi)
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
            if img.any():
                if (len(lastroi) > 0):
                    img = img[lastroi[0]:lastroi[1], lastroi[2]:lastroi[3]]
                cv2.imshow("Follow Face", img)
                cv2.resizeWindow("Follow Face", 320, 320)
    except:
        pass


if __name__ == '__main__':
    detectAndTrackLargestFace()
