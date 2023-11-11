########## Import ##########

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

########## Data ##########

video_path = "data/synthetic/escrime-4-3.avi"

########## Request function ##########

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

def initialize_tracking(video_capture):
  video_capture = cv2.VideoCapture(video_path)
  _,frame=video_capture.read()

  roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow("Select Object")

  video_capture.release()
  return roi

def histo_roi(roi, frame):
    x, y, w, h = roi
    image = cv2.imread(frame)

    tracked_area = image[y:y+h+20, x:x+w+20]
    hsv = cv2.cvtColor(tracked_area, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    return hist

def initialize_particles(roi, num_particles, sigma_rand=10):
    x, y, w, h = roi
    center=(int(x+w/2), int(y+h/2))

    paricles=np.array([np.random.normal(center, sigma_rand) for _ in range(num_particles)])
    weights=np.ones(num_particles)/num_particles

    return paricles, weights

def prediction_particle(roi, particles):
    x, y, w, h = roi
    list_pred_roi = []
    for particle in particles:
        x_part, y_part = particle
        roi_particle = int(x_part - w / 2), int(y_part - h / 2), w, h
        list_pred_roi.append(roi_particle)
    return list_pred_roi

def correction_particle(roi, frame, list_pred_roi, lamda = 5):
    hist_ref = histo_roi(roi, frame)
    list_weight = []
    for pred in list_pred_roi:
        hist_pred = histo_roi(pred, frame)
        dist = cv2.compareHist(hist_ref,hist_pred,cv2.HISTCMP_BHATTACHARYYA)
        likelihood = np.exp(-lamda*(dist**2))
        
    return list_weight

########## Plot function ##########

def show_square(roi,frame):
    im = Image.open(frame)
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    plt.show()

def show_hist(hist):
    plt.plot(hist)
    plt.title('Histogram of the Tracked Area')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

########## Tests ##########

extract_frame(video_path)

# For frame(0)

roi = [300,218,42,42]

particles, weight = initialize_particles(roi, 10)
list_pred_roi = prediction_particle(roi, particles)
print(correction_particle(roi, 'frames/frame_0.jpg', list_pred_roi))