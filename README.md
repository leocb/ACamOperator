# ACamOperator

The objective of this project is to point a camera at someone and by using OpenCV tracking systems, it'll automatically follow them with a pan and tilt mechanism

The intended use is for conferences, where you have to follow a person at all times, without losing the camera frame

This code would not exist without [Adrian Rosebrock and his tutorial](https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/). Thank you!

## Project specifics

This code is designed for the following specifications, your mileage may vary.

- This uses a EasyCap USB dongle to capture the image from a composite out of a camera
- EasyCaps are a little tricky to find a suitable driver and use within openCV, see bellow for more notes
- the output is trough a COM port, in this packet: `$[horizontalSpeedByte][VerticalSpeedByte][CS]`
- the `speed bytes` are a single char (signed), that is, 127 speeds available at either direction
- Checksum (CS) is a Xor of both bytes.

## Using

1. Connect the arduino to the PC
2. Run the `ACamOperator.py` script
3. Press `S` to select the object to be tracked, then press `Enter`
4. everything should work
5. press `Q` to quit.

if the `ACamOperator.py` script can't open the video stream (black screen), try running the `test_stream.py` first to check if everything is working.

## About EasyCap

- The Windows 10 driver that I use on my PC is in the `Drivers` folder, unzip and install.
- On openCV, use `cv2.VideoCapture` instead of `cv2.VideoStream`, and pass the `CAP_DSHOW` flag with it, then set the FPS. Example:

```python
cam = cv2.VideoCapture(streamIdHere + cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FPS, 30)
```