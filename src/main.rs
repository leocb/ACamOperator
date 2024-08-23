use anyhow::Result; // Automatically handle the error types
use opencv::{
    prelude::*,
    objdetect,
    imgproc,
    core,
    types,
    videoio,
    highgui
};
use opencv::core::Size;

// Note, the namespace of OpenCV is changed (to better or worse). It is no longer one enormous.
fn main() -> Result<()> { // Note, this is anyhow::Result
    // Open a GUI window
    highgui::named_window("ACamOperator", highgui::WINDOW_FULLSCREEN)?;
    // Open the web-camera (assuming you have one)
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let xml = "C:\\tools\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml";
    let mut face_detector = objdetect::CascadeClassifier::new(xml)?;
    let mut frame = Mat::default(); // This array will store the web-cam data
    // Read the camera
    // and display in the window
    loop {
        let mut cam_raw = Mat::default();
        cam.read(&mut cam_raw)?;
        imgproc::cvt_color(&cam_raw, &mut frame, imgproc::COLOR_RGB2GRAY, 0)?;
        let mut faces = types::VectorOfRect::new();
        face_detector.detect_multi_scale(
            &frame,
            &mut faces,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            Size::new(30,30),
            Size::new(0,0)
        )?;

        for face in faces.iter(){
            imgproc::rectangle(
                &mut cam_raw,
                face,
                core::Scalar::new(0f64,255f64,0f64,0f64),
                2,
                imgproc::LINE_8,
                0
            )?
        }

        highgui::imshow("ACamOperator", &cam_raw)?;
        let key = highgui::wait_key(1)?;
        if key == 113 { // quit with q
            break;
        }
    }
    Ok(())
}