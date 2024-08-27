# ACamOperator

The objective of this project is to point a camera at someone and by using OpenCV tracking systems, it'll automatically follow them with a pan and tilt mechanism

The intended use is for conferences, where you have to follow a person at all times, without losing the person from the frame

## Using

1. Download the files in the release.
2. Write the arduino to the board you're using.
3. Connect the arduino to the PC via USB (COM)
4. Run the `ACamOperator.exe`
5. Click on the detected face in the software to track it, if more faces are present, you can click on more than one.
6. To stop tracking a face, click on it again
7. Press the arrow keys `UP` or `DOWN` to increase or decrease the vertical offset
8. Press the arrow keys `LEFT` or `RIGHT` to increase or decrease the horizontal offset
9. everything should work
10. press and hold `Q` for 3 seconds to quit.

## Project specifics

This code is designed for the following specifications, your mileage may vary.

- This uses the webcam as input. In theory, anything that can be detected as a webcam can be used
- the output is through a COM port, in this packet format (not final): `$[horizontalSpeedByte][VerticalSpeedByte][CS]`
  - `$` is the start byte identifier
  - `speed bytes` are a single char (signed), that is, 127 speeds are available in either direction
  - `CS` is the checksum, an Xor of the 2 speed bytes.

## Developing

This project is developed using the open source license of Jetbrain's Rust Rover. I advise using it for editing this source code as well

### OpenCV binaries

- OpenCV Version used in this project: 4.10
- This project source does **NOT** include the OpenCV binaries. find out on your own how to obtain it and configure them on your system. Tutorial
  - https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md
- The OpenCV Binaries must be included in both the `target/debug` folder while developing and in the release distribution

### Yunet and sFace NN model weights

- This project uses yunet and sface for facial detection and recognition.
- A copy of the models weights are supplied with this project source in the `/models` folder.
- These models must be renamed to `yunet.onnx` and `sface.onnx` and copied to the `target/debug/models` folder for development
- For the release, these models must also be distributed with the final exe.
- The models are available in the [OpenCV/zoo](https://github.com/opencv/opencv_zoo/tree/main/models) repository and are under the Apache V2 License

## LICENSE SYNOPSIS

This code is licensed under the GPL3 License. TL;DR:

```markdown
1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. If you dare build your business solely from this code, you risk open-sourcing the whole code base.
6. If you modify it, you have to indicate changes made to the code.
7. Any modifications of this code base MUST be distributed with the same license, GPLv3.
8. This software is provided without warranty.
9. The software author or license can not be held liable for any damages inflicted by the software.
```

More information on about this license can be found [here](http://choosealicense.com/licenses/gpl-3.0/)