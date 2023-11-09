import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

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

roi = [300,218,42,42]

def show_square(roi,frame):
    im = Image.open(frame)
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)

    plt.show()

show_square(roi,'frames/frame_0.jpg')

def histo_roi(roi, frame):
    image = cv2.imread(frame)
    initial_area = (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3])

    tracked_area = image[roi[1]:roi[1]+roi[3]+20, roi[0]:roi[0]+roi[2]+20]
    hsv = cv2.cvtColor(tracked_area, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    plt.plot(hist)
    plt.title('Histogram of the Tracked Area')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    return hist

histo_roi(roi,'frames/frame_0.jpg')

def initialize_particles(roi, num_particles, sigma):
    x, y, w, h = roi
    center=(x+w//2, y+h//2)

    paricles=np.array([np.random.normal(center, sigma) for _ in range(num_particles)])
    weights=np.ones(num_particles)/num_particles

    noise = np.random.randn(particles.shape) * sigma
    particles += noise

    return paricles, weights