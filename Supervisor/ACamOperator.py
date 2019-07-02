# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

print("[INFO] Initializing everything")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,  # Accurate but slow
        "kcf": cv2.TrackerKCF_create,  # Mid accuracy and speed
        # fast but can't handle rapid changes
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create  # FAST
    }

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(2 + cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FPS, 30)
print("[INFO] video stream is open")

# initialize the FPS throughput estimator
fps = None

# loop over frames from the video stream
while True:

    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    # frame = frame[1] if args.get("video", False) else frame

    pframe = frame.copy()

    # De-interlace
    pframe[1::2] = pframe[::2]

    # check to see if we have reached the end of the stream

    # resize the frame (so we can process it faster) and grab the frame dimensions
    # frame = imutils.resize(frame, width=500)
    (H, W) = pframe.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(pframe)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(pframe, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        else:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(pframe, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)

        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(pframe, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Tracking", pframe)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Tracking", pframe, fromCenter=False,
                               showCrosshair=True)
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(pframe, initBB)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

print("[INFO] Destroy all")
cam.release()
cv2.destroyAllWindows()
