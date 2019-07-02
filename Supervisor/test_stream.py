# import the necessary packages
import cv2

cam = cv2.VideoCapture(2 + cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FPS, 29.97)

while(True):
    ret, frame = cam.read()
    cv2.imshow('frame', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
