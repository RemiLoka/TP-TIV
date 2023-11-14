########## Import ##########

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

########## Data ##########

video_path = "data/synthetic/escrime-4-3.avi"
cap = cv2.VideoCapture(video_path)

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

def initialize_tracking(video_path):
  video_capture = cv2.VideoCapture(video_path)
  _,frame=video_capture.read()

  roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow("Select Object")

  video_capture.release()
  return roi

def histo_roi(roi, image):
    x, y, w, h = roi

    tracked_area = image[y:y+h+20, x:x+w+20]
    hsv = cv2.cvtColor(tracked_area, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    return hist

def initialize_particles(roi, num_particles, sigma_rand= 20):
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

def correction_particle(roi, frame, list_pred_roi, weights, lamda = 5):
    hist_ref = histo_roi(roi, frame)
    for i in range(len(list_pred_roi)):
        hist_pred = histo_roi(list_pred_roi[i], frame)
        dist = cv2.compareHist(hist_ref,hist_pred,cv2.HISTCMP_BHATTACHARYYA)
        weights[i] = np.exp(-lamda*(dist**2))
    weights /= np.sum(weights)
    return weights

def resampling_particle(particles, weights):
    N = len(weights)
    indices = []

    cumulative_weights = np.cumsum(weights)
    u = np.random.uniform(0, 1 / N)

    j = 0
    for i in range(N):
        while u + i / N > cumulative_weights[j]:
            j += 1
        indices.append(j)
    selected_particles = [particles[i] for i in indices]
    selected_weights = [weights[i] for i in indices]
    return selected_particles, selected_weights

def next_pos(selected_particles,selected_weights):
    return np.average(selected_particles, weights = selected_weights, axis=0)

def visualize_tracking(frame, pos_estimate, roi):
    x, y = pos_estimate
    w, h = roi[2:]
    top_left = (int(x - w / 2), int(y - h / 2))
    bottom_right = (int(x + w / 2), int(y + h / 2))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    particles, weights = initialize_particles(roi, 30)
    list_pred_roi = prediction_particle(roi, particles)
    weights = correction_particle(roi, frame, list_pred_roi, weights)
    selected_particles, selected_weights = resampling_particle(list_pred_roi, weights)
    estimate_pos = next_pos(selected_particles, selected_weights)
    visualize_tracking(frame,estimate_pos,roi)
    cv2.imshow('frame', frame)
cap.release()

particles, weights = initialize_particles(roi, 30)
list_pred_roi = prediction_particle(roi, particles)
weights = correction_particle(roi, 'frames/frame_0.jpg', list_pred_roi, weights)
print(resampling_particle(list_pred_roi,weights))