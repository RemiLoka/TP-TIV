import cv2
import os

def extract_frame(video_path):
    video_capture = cv2.VideoCapture(video_path)

    output_folder = 'frames'
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_path = f'{output_folder}/frame_{frame_count}.jpg'
        cv2.imwrite(frame_path, frame) 

        frame_count += 1

    video_capture.release()

video_path = "data/synthetic/escrime-4-3.avi"
extract_frame(video_path)

def initialize_tracking(video_capture):
  video_capture = cv2.VideoCapture(video_path)
  _,frame=video_capture.read()

  roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow("Select Object")

  video_capture.release()
  return roi

roi = initialize_tracking(video_path)

print(roi)