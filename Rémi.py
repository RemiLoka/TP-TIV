########## Import ##########

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

########## Data ##########

video_path = 'data/synthetic/escrime-4-3.avi'
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

def initialize_tracking(video_capture):
  _,frame=video_capture.read()

  roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow("Select Object")
  return roi

def histo_roi(roi, image):
    x, y, w, h = roi

    tracked_area = image[y:y+h, x:x+w]

    hist_channels = [cv2.calcHist([tracked_area], [i], None, [256], [0, 256]) for i in range(3)]
    for i in range(3):
        cv2.normalize(hist_channels[i], hist_channels[i], 0, 255, cv2.NORM_MINMAX)
    hist = np.concatenate(hist_channels, axis=None)
    return hist

def initialize_particles(roi, num_particles, sigma_rand= 5):
    center = roi[:2]
    particles=np.array([np.random.normal(center, sigma_rand) for _ in range(num_particles)])
    weights=np.ones(num_particles)/num_particles
    return particles, weights

def prediction_particle(roi, particles):
    w, h = roi[2:]
    list_pred_roi = []
    for particle in particles:
        x_part, y_part = particle
        roi_particle = int(x_part - w // 2), int(y_part - h // 2), w, h
        list_pred_roi.append(roi_particle)
    return list_pred_roi

def correction_particle(roi, frame, list_pred_roi, weights, lamda = 1):
    hist_ref = histo_roi(roi, frame)
    for i in range(len(list_pred_roi)):
        hist_pred = histo_roi(list_pred_roi[i], frame)
        dist = cv2.compareHist(hist_ref,hist_pred,cv2.HISTCMP_BHATTACHARYYA)
        weights[i] = np.exp(-lamda*(dist**2))
    weights /= np.sum(weights)
    return weights

def resampling_particle(particles, weights):
    N = len(weights)
    selected_particles = []

    cumulative_weights = np.cumsum(weights)
    u = np.random.uniform(0, 1 / N)

    j = 0
    for i in range(N):
        while u + (i / N) > cumulative_weights[j]:
            j += 1
        selected_particles.append(particles[j][:])
    return selected_particles

def next_pos(selected_particles,selected_weights):
    return np.average(selected_particles, weights = selected_weights, axis=0)

def apply_roi(estimate_pos, roi):
    x, y = estimate_pos[:2]
    w, h = roi[2:]
    return x, y, w, h

def visualize_tracking(frame, particles, pos_estimate, roi):
    x, y = pos_estimate[:2]
    w, h = roi[2:]
    top_left = (int(x - w / 2), int(y - h / 2))
    bottom_right = (int(x + w / 2), int(y + h / 2))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    for particle in particles:
        px, py = int(particle[0]), int(particle[1])
        cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

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

roi = initialize_tracking(cap)

#roi = [300,218,42,42]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    particles, weights = initialize_particles(roi, 20)
    list_pred_roi = prediction_particle(roi, particles)
    weights = correction_particle(roi, frame, list_pred_roi, weights)
    selected_particles = resampling_particle(particles, weights)
    estimate_pos = next_pos(selected_particles, weights)
    roi = apply_roi(estimate_pos, roi)
    visualize_tracking(frame,selected_particles,estimate_pos,roi)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()

# particles, weights = initialize_particles(roi, 30)
# list_pred_roi = prediction_particle(roi, particles)
# weights = correction_particle(roi, 'frames/frame_0.jpg', list_pred_roi, weights)
# print(resampling_particle(list_pred_roi,weights))