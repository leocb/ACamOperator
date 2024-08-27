use anyhow::{Error, Result};
use opencv::core::{Scalar, Size, TickMeter};
use opencv::objdetect::FaceDetectorYN;
use opencv::videoio::{CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH};
// Automatically handle the error types
use opencv::{
    core,
    highgui,
    imgproc,
    prelude::*,
    videoio,
};


fn visualize(input: &mut Mat, faces: &Mat, fps: f64, thickness: i32) -> Result<()> {
    let fps_string = format!("FPS : {:.2}", fps);
    for i in 0..faces.rows() {
        // Draw bounding box
        let x = *faces.at_2d::<f32>(i, 0)?;
        let y = *faces.at_2d::<f32>(i, 1)?;
        let w = *faces.at_2d::<f32>(i, 2)?;
        let h = *faces.at_2d::<f32>(i, 3)?;
        let score = *faces.at_2d::<f32>(i, 14)?;
        let rect = core::Rect2f::new(x, y, w, h).to::<i32>().unwrap();
        imgproc::rectangle(input, rect, (0., 255., 0.).into(), thickness, imgproc::LINE_8, 0)?;

        // info text
        imgproc::put_text(
            input,
            format!("score: {score:.2}").as_str(),
            core::Point::new(x as i32, y as i32 - 5),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            (0., 255., 0.).into(),
            thickness,
            imgproc::LINE_8,
            false,
        )?;

        // Draw landmarks
        visualize_draw_point(input, faces, thickness, i, (255., 0., 0.).into(), 4)?;
        visualize_draw_point(input, faces, thickness, i, (0., 0., 255.).into(), 6)?;
        visualize_draw_point(input, faces, thickness, i, (0., 255., 0.).into(), 8)?;
        visualize_draw_point(input, faces, thickness, i, (255., 0., 255.).into(), 10)?;
        visualize_draw_point(input, faces, thickness, i, (0., 255., 255.).into(), 12)?;
    }

    // FPS
    imgproc::put_text(
        input,
        &fps_string,
        core::Point::new(0, 15),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        (0., 255., 0.).into(),
        thickness,
        imgproc::LINE_8,
        false,
    )?;
    Ok(())
}

fn visualize_draw_point(input: &mut Mat, faces: &Mat, thickness: i32, i: i32, color: Scalar, landmark_offset: i32) -> Result<(), Error> {
    imgproc::circle(
        input,
        core::Point2f::new(
            *faces.at_2d::<f32>(i, landmark_offset)?,
            *faces.at_2d::<f32>(i, landmark_offset + 1)?,
        ).to::<i32>().unwrap(),
        2,
        color,
        thickness,
        imgproc::LINE_8,
        0,
    )?;
    Ok(())
}

// Note, the namespace of OpenCV is changed (to better or worse). It is no longer one enormous.
fn main() -> Result<()> { // Note, this is anyhow::Result

    // Webcam/Input resolution
    let width = 1920.;
    let height = 1080.;

    // Open a GUI window
    highgui::named_window("ACamOperator", highgui::WINDOW_NORMAL)?;
    // Open the web-camera (assuming you have one)
    let mut cam = videoio::VideoCapture::new(2, videoio::CAP_ANY)?;
    cam.set(CAP_PROP_FRAME_WIDTH, width)?;
    cam.set(CAP_PROP_FRAME_HEIGHT, height)?;

    //timer (for fps)
    let mut tm = TickMeter::default()?;

    // Create the Yunet face detection and load NN weights
    let current_dir = std::env::current_dir().unwrap();
    let yunet_weights_path = current_dir.join("models").join("yunet.onnx").into_os_string().into_string().unwrap();
    let mut face_detector = FaceDetectorYN::create(
        &yunet_weights_path,
        "",
        Size::new(width as i32, height as i32),
        0.8f32,
        0.3f32,
        5000,
        0,
        0)?;


    loop {
        tm.start()?;
        // Read the camera
        let mut cam_raw = Mat::default();
        cam.read(&mut cam_raw)?;
        // convert the camera format?
        // imgproc::cvt_color(&cam_raw, &mut frame, imgproc::COLOR_RGB2GRAY, 0)?;

        // Detect faces
        let mut faces = Mat::default();
        face_detector.detect(&cam_raw, &mut faces)?;

        let mut output_image = cam_raw.clone();
        tm.stop()?;

        visualize(&mut output_image, &faces, tm.get_fps()?, 2)?;

        // and display in the window
        highgui::imshow("ACamOperator", &output_image)?;
        let key = highgui::wait_key(1)?;
        if key == 113 { // quit with q
            break;
        }
    }
    Ok(())
}