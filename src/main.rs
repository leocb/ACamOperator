use anyhow::{Error, Result};
use opencv::core::{Scalar, Size, TickMeter, Vector};
use opencv::highgui::MouseEventTypes::{EVENT_LBUTTONDOWN, EVENT_LBUTTONUP};
use opencv::imgproc::INTER_AREA;
use opencv::objdetect::{FaceDetectorYN, FaceRecognizerSF, FaceRecognizerSF_DisType};
use opencv::videoio::{CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH};
// Automatically handle the error types
use opencv::{
    core,
    highgui,
    imgproc,
    prelude::*,
    videoio,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

fn visualize(input: &mut Mat, faces: &Mat, matches: &Vec<usize>, fps: f64, thickness: i32) -> Result<()> {
    let fps_string = format!("FPS : {:.2}", fps);
    let mut j = 0;
    for i in 0..faces.rows() {

        // extract data
        let x = *faces.at_2d::<f32>(i, 0)?;
        let y = *faces.at_2d::<f32>(i, 1)?;
        let w = *faces.at_2d::<f32>(i, 2)?;
        let h = *faces.at_2d::<f32>(i, 3)?;

        let match_id = matches[j];
        //let score = *faces.at_2d::<f32>(i, 14)?;

        // bounding box
        let rect = core::Rect2f::new(x, y, w, h).to::<i32>().unwrap();
        imgproc::rectangle(
            input,
            rect,
            if match_id == 99 { (0., 255., 0.).into() } else { (0., 0., 255.).into() },
            thickness,
            imgproc::LINE_8,
            0)?;

        // info text
        imgproc::put_text(
            input,
            format!("ID: {match_id}").as_str(),
            core::Point::new(x as i32, y as i32 - 5),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            if match_id == 99 { (0., 255., 0.).into() } else { (0., 0., 255.).into() },
            thickness,
            imgproc::LINE_8,
            false,
        )?;
        j += 1;

        // Draw landmarks
        visualize_draw_point(input, &faces, thickness, (255., 0., 0.).into(), i, 4)?;
        visualize_draw_point(input, &faces, thickness, (0., 0., 255.).into(), i, 6)?;
        visualize_draw_point(input, &faces, thickness, (0., 255., 0.).into(), i, 8)?;
        visualize_draw_point(input, &faces, thickness, (255., 0., 255.).into(), i, 10)?;
        visualize_draw_point(input, &faces, thickness, (0., 255., 255.).into(), i, 12)?;
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

fn visualize_draw_point(input: &mut Mat, faces: &Mat, thickness: i32, color: Scalar, i: i32, landmark_offset: i32) -> Result<(), Error> {
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

#[derive(Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
}
impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

// Note, the namespace of OpenCV is changed (to better or worse). It is no longer one enormous.
fn main() -> Result<()> { // Note, this is anyhow::Result

    // Webcam ID / Resolution
    let cam_id = 2;
    let cam_width = 1920.;
    let cam_height = 1080.;
    let yunet_width = 600.; // 2x 300px (yunet max res)
    let scale = cam_width / yunet_width;
    let yunet_height = cam_height / scale;

    // Open a GUI window
    highgui::named_window("ACamOperator", highgui::WINDOW_NORMAL)?;
    // Open the web-camera (assuming you have one)
    let mut cam = videoio::VideoCapture::new(cam_id, videoio::CAP_ANY)?;
    cam.set(CAP_PROP_FRAME_WIDTH, cam_width)?;
    cam.set(CAP_PROP_FRAME_HEIGHT, cam_height)?;

    // mouse events
    let mut mouse_pos = Point::new(-1, -1);
    let mut instant_click = false;

    let default_mouse_event_data = (EVENT_LBUTTONDOWN, 0, 0, 0);
    let mouse_event_data = Arc::new(Mutex::new(default_mouse_event_data));
    let should_handle_mouse_event = Arc::new(AtomicBool::new(false));
    let mouse_event_dispatcher = {
        let mouse_data = Arc::clone(&mouse_event_data);
        let should_handle_mouse_event = Arc::clone(&should_handle_mouse_event);

        move |event: i32, x: i32, y: i32, flags: i32| {
            // can intercept specific mouse events here to don't update the mouse_data
            if let Ok(mouse_event) = highgui::MouseEventTypes::try_from(event) {
                if let Ok(mut mouse_data) = mouse_data.lock() {
                    *mouse_data = (mouse_event, x, y, flags);
                }
            }
            should_handle_mouse_event.store(true, Ordering::Relaxed);
        }
    };
    highgui::set_mouse_callback("ACamOperator", Some(Box::new(mouse_event_dispatcher))).expect("Cannot set mouse callback");


    //timer (for fps)
    let mut fps = TickMeter::default()?;

    // Create the Yunet face detection and load NN weights
    let current_dir = std::env::current_dir().unwrap();
    let yunet_weights_path = current_dir.join("models").join("yunet.onnx").into_os_string().into_string().unwrap();
    let mut face_detector = FaceDetectorYN::create(
        &yunet_weights_path,
        "",
        Size::new(yunet_width as i32, yunet_height as i32),
        0.8f32,
        0.3f32,
        5000,
        0,
        0)?;

    // Create the sface recognition and load NN weights
    let sface_weights_path = current_dir.join("models").join("sface.onnx").into_os_string().into_string().unwrap();
    let mut face_recognizer = FaceRecognizerSF::create_def(&sface_weights_path, "")?;

    let mut saved_faces_features: Vector<Mat> = Vector::new();
    let cosine_similar_thresh = 0.363;
    let l2norm_similar_thresh = 1.128;

    loop {
        fps.reset()?;

        // Mouse events
        let (mouse_event, mouse_x, mouse_y, _) = {
            if should_handle_mouse_event.load(Ordering::Relaxed) {
                should_handle_mouse_event.store(false, Ordering::Relaxed);
                if let Ok(mouse_event_data) = mouse_event_data.lock() {
                    *mouse_event_data
                } else {
                    default_mouse_event_data
                }
            } else {
                default_mouse_event_data
            }
        };

        if mouse_event == EVENT_LBUTTONDOWN {
            mouse_pos = Point::new(mouse_x, mouse_y);
            instant_click = true;
        }

        if mouse_event == EVENT_LBUTTONUP {
            mouse_pos = Point::new(-1, -1);
            instant_click = false;
        }

        // Read the camera
        fps.start()?;
        let mut cam_raw = Mat::default();
        cam.read(&mut cam_raw)?;
        // convert the camera format?
        // imgproc::cvt_color(&cam_raw, &mut frame, imgproc::COLOR_RGB2GRAY, 0)?;

        if cam_raw.size()?.width <= 0 || cam_raw.size()?.height <= 0 {
            continue;
        }

        // Scale image
        let mut cam_scaled = Mat::default();
        imgproc::resize(&cam_raw, &mut cam_scaled, Size::new(yunet_width as i32, yunet_height as i32), 0., 0., INTER_AREA)?;

        // Detect faces
        let mut faces_low_res = Mat::default();
        face_detector.detect(&cam_scaled, &mut faces_low_res)?;

        // Scale faces found to full res image coordinates
        let mut faces = Mat::default();
        faces_low_res.convert_to(&mut faces, -1, scale, 0.)?;

        // Recognize faces
        let mut matches: Vec<usize> = Vec::new();
        for i in 0..faces.rows() {

            // click inside a box
            let mut is_inside = false;
            let x = *faces.at_2d::<f32>(i, 0)? as i32;
            let y = *faces.at_2d::<f32>(i, 1)? as i32;
            let w = *faces.at_2d::<f32>(i, 2)? as i32;
            let h = *faces.at_2d::<f32>(i, 3)? as i32;
            if instant_click {
                if mouse_pos.x >= x && mouse_pos.x <= x + w &&
                    mouse_pos.y >= y && mouse_pos.y <= y + h {
                    is_inside = true;
                }
            }

            // grab the face crop
            let mut face_crop = Mat::default();
            face_recognizer.align_crop(&cam_raw, &faces.row(i)?, &mut face_crop)?;

            // Run feature extraction with given aligned_face
            let mut features = Mat::default();
            face_recognizer.feature(&face_crop, &mut features)?;


            // match
            let mut matched: usize = 99;
            for (j, saved_feature) in saved_faces_features.iter().enumerate() {
                let score_cos = face_recognizer.match_(&features, &saved_feature, FaceRecognizerSF_DisType::FR_COSINE.into())?;
                let score_l2 = face_recognizer.match_(&features, &saved_feature, FaceRecognizerSF_DisType::FR_NORM_L2.into())?;

                if score_cos > cosine_similar_thresh || score_l2 <= l2norm_similar_thresh {

                    // clicked again, remove from list
                    if instant_click && is_inside {
                        saved_faces_features.remove(j)?;
                        instant_click = false;
                        break;
                    }

                    // get the id of the face
                    matched = j;
                    break;
                }
            }

            matches.push(matched);


            // Save features if clicked (follow on next frame)
            if instant_click && is_inside {
                saved_faces_features.push(features.try_clone()?);
                instant_click = false;
            }
        }
        fps.stop()?;

        // Visualize
        visualize(&mut cam_raw, &faces, &matches, fps.get_fps()?, 2)?;

        // display in the window
        if cam_raw.rows() > 0 && cam_raw.cols() > 0 {
            highgui::imshow("ACamOperator", &cam_raw)?;
        }

        // quit with "q"
        let key = highgui::wait_key(1)?;
        if key == 113 { // quit with q
            break;
        }
    }
    Ok(())
}