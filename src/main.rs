use anyhow::{Error, Result};
use opencv::core::{Point, Scalar, Size, TickMeter, Vector};
use opencv::highgui::MouseEventTypes::{EVENT_LBUTTONDOWN, EVENT_LBUTTONUP};
use opencv::imgproc::{line, INTER_AREA, LINE_8};
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
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

fn visualize(input: &mut Mat, faces: &Mat, matches: &Vec<usize>, fps: f64, trails: &Vec<AllocRingBuffer<Point>>, target: &Point, track_center: &Point) -> Result<()> {
    let thickness = 2;
    let fps_string = format!("FPS : {:.2}", fps);
    let mut j = 0;

    // rule of thirds
    draw_crosshair_at_point(input, Point::new(input.size().unwrap().width / 3, input.size().unwrap().height / 3), (255., 255., 255.).into(), thickness)?;
    draw_crosshair_at_point(input, Point::new(input.size().unwrap().width / 3 * 2, input.size().unwrap().height / 3 * 2), (255., 255., 255.).into(), thickness)?;

    // target lines
    draw_crosshair_at_point(input, *target, (255., 0., 0.).into(), thickness)?;

    // Track and distance to center
    if track_center.x > 0 {
        draw_point(input, *track_center, (0., 0., 255.).into(), 10)?;
        line(input, *track_center, *target, (0., 255., 255.).into(), thickness, LINE_8, 0)?;
    }

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
            if match_id == 99 { (255., 255., 255.).into() } else { (0., 0., 255.).into() },
            thickness,
            imgproc::LINE_8,
            0)?;

        // Movement trail
        if match_id != 99 {
            let trail_vec = trails[match_id].to_vec();
            for t in 1..trail_vec.len() {
                line(input, trail_vec[t], trail_vec[t - 1], (0., 0., 255.).into(), thickness, LINE_8, 0)?;
            }

            // info text
            imgproc::put_text(
                input,
                format!("ID: {match_id}").as_str(),
                core::Point::new(x as i32, y as i32 - 5),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                (0., 0., 255.).into(),
                thickness,
                imgproc::LINE_8,
                false,
            )?;
        }

        // Draw landmarks
        //draw_face_landmarks(input, &faces, thickness, i)?;

        j += 1;
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

fn draw_face_landmarks(input: &mut Mat, faces: &&Mat, thickness: i32, i: i32) -> Result<(), Error> {
    visualize_face_features_point(input, &faces, thickness, (255., 0., 0.).into(), i, 4)?;
    visualize_face_features_point(input, &faces, thickness, (0., 0., 255.).into(), i, 6)?;
    visualize_face_features_point(input, &faces, thickness, (0., 255., 0.).into(), i, 8)?;
    visualize_face_features_point(input, &faces, thickness, (255., 0., 255.).into(), i, 10)?;
    visualize_face_features_point(input, &faces, thickness, (0., 255., 255.).into(), i, 12)?;
    Ok(())
}

fn visualize_face_features_point(input: &mut Mat, faces: &Mat, thickness: i32, color: Scalar, i: i32, landmark_offset: i32) -> Result<(), Error> {
    draw_point(input, core::Point2f::new(*faces.at_2d::<f32>(i, landmark_offset)?, *faces.at_2d::<f32>(i, landmark_offset + 1)?,
    ).to::<i32>().unwrap(), color, thickness)?;
    Ok(())
}

fn draw_crosshair_at_point(input: &mut Mat, point: Point, color: Scalar, thickness: i32) -> Result<(), Error> {
    line(input, Point::new(point.x, 0), Point::new(point.x, input.size().unwrap().height), color, thickness, LINE_8, 0)?;
    line(input, Point::new(0, point.y), Point::new(input.size().unwrap().width, point.y), color, thickness, LINE_8, 0)?;
    Ok(())
}

fn draw_point(input: &mut Mat, point: Point, color: Scalar, thickness: i32) -> Result<(), Error> {
    imgproc::circle(input, point, 2, color, thickness, LINE_8, 0)?;
    Ok(())
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

    // Visualization Trail
    let mut trails: Vec<AllocRingBuffer<Point>> = Vec::new();

    // Global Target
    let mut global_target = Point::new((cam_width / 2.) as i32, (cam_height / 3.5) as i32);
    let mut tracking_center = Point::new(0, 0);
    let mut expected_match_count = 0;

    // Open a GUI window
    highgui::named_window("ACamOperator", highgui::WINDOW_NORMAL)?;

    // Open the web-camera (assuming you have one)
    let mut cam = videoio::VideoCapture::new(cam_id, videoio::CAP_ANY)?;
    cam.set(CAP_PROP_FRAME_WIDTH, cam_width)?;
    cam.set(CAP_PROP_FRAME_HEIGHT, cam_height)?;

    // mouse events
    let mut mouse_pos = Point::new(-1, -1);
    let mut mouse_down = false;

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
        // keep counter up to date
        fps.reset()?;

        // face id tracking deletion
        let mut stop_tracking_id: usize = 99;

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
            mouse_down = true;
        }

        if mouse_event == EVENT_LBUTTONUP {
            mouse_pos = Point::new(-1, -1);
            mouse_down = false;
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
        let mut match_count = 0;
        tracking_center = Point::new(0, 0);
        for i in 0..faces.rows() {

            // click inside a box
            let mut is_mouse_inside = false;
            let x = *faces.at_2d::<f32>(i, 0)? as i32;
            let y = *faces.at_2d::<f32>(i, 1)? as i32;
            let w = *faces.at_2d::<f32>(i, 2)? as i32;
            let h = *faces.at_2d::<f32>(i, 3)? as i32;
            let mid_x = x + w / 2;
            let mid_y = y + h / 2;
            if mouse_down {
                if mouse_pos.x >= x && mouse_pos.x <= x + w &&
                    mouse_pos.y >= y && mouse_pos.y <= y + h {
                    is_mouse_inside = true;
                }
            }

            // grab the face crop
            let mut face_crop = Mat::default();
            face_recognizer.align_crop(&cam_raw, &faces.row(i)?, &mut face_crop)?;

            // Run feature extraction with given aligned_face
            let mut features = Mat::default();
            face_recognizer.feature(&face_crop, &mut features)?;


            // match
            let mut match_id: usize = 99;
            for (j, saved_feature) in saved_faces_features.iter().enumerate() {
                let score_cos = face_recognizer.match_(&features, &saved_feature, FaceRecognizerSF_DisType::FR_COSINE.into())?;
                let score_l2 = face_recognizer.match_(&features, &saved_feature, FaceRecognizerSF_DisType::FR_NORM_L2.into())?;

                if score_cos > cosine_similar_thresh || score_l2 <= l2norm_similar_thresh {

                    // clicked on tracked face, mark it for stop tracking
                    if mouse_down && is_mouse_inside {
                        stop_tracking_id = j;
                        mouse_down = false;
                        break;
                    }

                    // get the id of the face
                    match_id = j;
                    // trail
                    trails[match_id].push(Point::new(mid_x, mid_y));
                    // track center
                    match_count += 1;
                    tracking_center.x += mid_x;
                    tracking_center.y += mid_y;
                    break;
                }
            }

            matches.push(match_id);


            // Save features if clicked (follow on next frame)
            if mouse_down && is_mouse_inside {
                expected_match_count += 1;
                saved_faces_features.push(features.try_clone()?);
                let mut new_trail_buffer = AllocRingBuffer::new(30);
                new_trail_buffer.fill(Point::new(mid_x, mid_y));
                trails.push(new_trail_buffer);
                mouse_down = false;
            }
        }

        // get the center of all tracked faces
        if match_count > 0 {
            tracking_center.x /= match_count;
            tracking_center.y /= match_count;
        }

        // stop moving if cant find all tracked faces
        if match_count < expected_match_count {
            tracking_center.x = 0;
            tracking_center.y = 0;
        }

        // count the fps until here
        fps.stop()?;

        // Visualize
        visualize(&mut cam_raw, &faces, &matches, fps.get_fps()?, &trails, &global_target, &tracking_center)?;

        // Remove tracking if necessary
        if stop_tracking_id != 99 {
            saved_faces_features.remove(stop_tracking_id)?;
            trails.remove(stop_tracking_id);
            expected_match_count -= 1;
        }

        // display in the window
        if cam_raw.rows() > 0 && cam_raw.cols() > 0 {
            highgui::imshow("ACamOperator", &cam_raw)?;
        }

        // keyboard handle.
        // - Q: Quit
        // - C: Clear all tracking
        // - Arrows: Reposition global target
        // - Center (numpad 5) - reset target
        let key = highgui::wait_key(1)?;
        match key {
            50 => { // Down (numpad 2)
                global_target.y += 5;
            }
            56 => { // Up (numpad 8)
                global_target.y -= 5;
            }
            52 => { // Left (numpad 4)
                global_target.x -= 5;
            }
            54 => { // Right (numpad 6)
                global_target.x += 5;
            }
            53 => { // Center (numpad 5) - reset target
                global_target = Point::new((cam_width / 2.) as i32, (cam_height / 3.5) as i32);
            }
            99 => { // C - Clear all tracking
                saved_faces_features.clear();
                trails.clear();
                expected_match_count = 0;
            }
            113 => { // Q - Quit
                break;
            }
            _ => ()
        }
    }
    Ok(())
}