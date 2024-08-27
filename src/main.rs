use std::path::Path;
use anyhow::{Error, Result}; // Automatically handle the error types
use opencv::{
    prelude::*,
    objdetect,
    imgproc,
    core,
    videoio,
    highgui,
};
use opencv::core::{float16_t, MatIter, Rect, Scalar, Size, TickMeter, Vector};
use opencv::objdetect::FaceDetectorYN;
use opencv::sys::{cv_FaceDetectorYN_create_const_StringR_const_StringR_const_SizeR, cv_FaceDetectorYN_create_const_StringR_const_StringR_const_SizeR_float_float_int_int_int};
use opencv::videoio::{CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH};


fn visualize(input: &mut Mat, frame: i32, faces: &Mat, fps: f64, thickness: i32) -> anyhow::Result<()> {
    let fps_string = format!("FPS : {:.2}", fps);
    if frame >= 0 {
        println!("Frame {}, ", frame);
    }
    println!("FPS: {}", fps_string);
    for i in 0..faces.rows() {
        // Print results
        println!(
            "Face {i}, top-left coordinates: ({}, {}), box width: {}, box height: {}, score: {:.2}",
            faces.at_2d::<f32>(i, 0)?,
            faces.at_2d::<f32>(i, 1)?,
            faces.at_2d::<f32>(i, 2)?,
            faces.at_2d::<f32>(i, 3)?,
            faces.at_2d::<f32>(i, 14)?
        );

        // Draw bounding box
        let rect = core::Rect2f::new(
            *faces.at_2d::<f32>(i, 0)?,
            *faces.at_2d::<f32>(i, 1)?,
            *faces.at_2d::<f32>(i, 2)?,
            *faces.at_2d::<f32>(i, 3)?,
        )
            .to::<i32>()
            .ok_or_else(|| Error::msg("Invalid rect"))?;
        imgproc::rectangle(input, rect, (0., 255., 0.).into(), thickness, imgproc::LINE_8, 0)?;
        // Draw landmarks
        imgproc::circle(
            input,
            core::Point2f::new(*faces.at_2d::<f32>(i, 4)?, *faces.at_2d::<f32>(i, 5)?)
                .to::<i32>()
                .ok_or_else(|| Error::msg("Invalid point"))?,
            2,
            (255., 0., 0.).into(),
            thickness,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::circle(
            input,
            core::Point2f::new(*faces.at_2d::<f32>(i, 6)?, *faces.at_2d::<f32>(i, 7)?)
                .to::<i32>()
                .ok_or_else(|| Error::msg("Invalid point"))?,
            2,
            (0., 0., 255.).into(),
            thickness,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::circle(
            input,
            core::Point2f::new(*faces.at_2d::<f32>(i, 8)?, *faces.at_2d::<f32>(i, 9)?)
                .to::<i32>()
                .ok_or_else(|| Error::msg("Invalid point"))?,
            2,
            (0., 255., 0.).into(),
            thickness,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::circle(
            input,
            core::Point2f::new(*faces.at_2d::<f32>(i, 10)?, *faces.at_2d::<f32>(i, 11)?)
                .to::<i32>()
                .ok_or_else(|| Error::msg("Invalid point"))?,
            2,
            (255., 0., 255.).into(),
            thickness,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::circle(
            input,
            core::Point2f::new(*faces.at_2d::<f32>(i, 12)?, *faces.at_2d::<f32>(i, 13)?)
                .to::<i32>()
                .ok_or_else(|| Error::msg("Invalid point"))?,
            2,
            (0., 255., 255.).into(),
            thickness,
            imgproc::LINE_8,
            0,
        )?;
    }
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

// Note, the namespace of OpenCV is changed (to better or worse). It is no longer one enormous.
fn main() -> Result<()> { // Note, this is anyhow::Result

    // Webcam/Input resolution
    let width = 1280.;
    let height = 720.;

    // Open a GUI window
    highgui::named_window("ACamOperator", highgui::WINDOW_NORMAL)?;
    // Open the web-camera (assuming you have one)
    let mut cam = videoio::VideoCapture::new(1, videoio::CAP_DSHOW)?;
    cam.set(CAP_PROP_FRAME_WIDTH, width)?;
    cam.set(CAP_PROP_FRAME_HEIGHT, height)?;

    //timer (for fps)
    let mut tm = TickMeter::default()?;

    // Create the Yunet face detection and load NN weights
    let yunet_weights_path = std::env::current_dir?.join("models").join("yunet.onnx").into_os_string().into_string()?;
    let mut face_detector = FaceDetectorYN::create(
        yunet_weights_path,
        "",
        Size::new(width as i32, height as i32),
        0.8f32,
        0.3f32,
        5000,
        0,
        0)?;



    // Visualization colors
    let box_color: Scalar = Scalar::new(0f64, 255f64, 0f64, 255f64);
    let mut landmark_color:Vector<Scalar> = Vector::new();
    landmark_color.push(Scalar::new(255f64,   0f64,   0f64, 255f64)); // right eye
    landmark_color.push(Scalar::new(  0f64,   0f64, 255f64, 255f64)); // left eye
    landmark_color.push(Scalar::new(  0f64, 255f64,   0f64, 255f64)); // nose tip
    landmark_color.push(Scalar::new(255f64,   0f64, 255f64, 255f64)); // right mouth corner
    landmark_color.push(Scalar::new(  0f64, 255f64, 255f64, 255f64)); // left mouth corner
    let text_color: Scalar =  Scalar::new(0f64, 255f64, 0f64, 255f64);

    loop {
        tm.start()?;
        // Read the camera
        let mut cam_raw = Mat::default();
        cam.read(&mut cam_raw)?;
        // convert the camera format?
        // imgproc::cvt_color(&cam_raw, &mut frame, imgproc::COLOR_RGB2GRAY, 0)?;

        // Detect faces
        let mut faces = Mat::default();
        face_detector.detect(&cam_raw,&mut faces)?;

        let output_image = cam_raw.clone();
        tm.stop()?;

        // if (fps >= 0)
        // {
        //     cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
        // }

        for i in 0..faces.rows()
        {
            visualize(&mut cam_raw, -1, &faces, tm.get_fps()?, 2)?;

            //
            // // Confidence as text
            // float conf = faces.at<float>(i, 14);
            // cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1+12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);
            //
            // // Draw landmarks
            // for (int j = 0; j < landmark_color.size(); ++j)
            // {
            //     int x = static_cast<int>(faces.at<float>(i, 2*j+4)), y = static_cast<int>(faces.at<float>(i, 2*j+5));
            //     cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
            // }
        }

        // and display in the window
        highgui::imshow("ACamOperator", &cam_raw)?;
        let key = highgui::wait_key(1)?;
        if key == 113 { // quit with q
            break;
        }
    }
    Ok(())
}