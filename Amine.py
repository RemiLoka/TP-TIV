########## Import ##########


import cv2
import numpy as np
import matplotlib.pyplot as plt


########## Data ##########


cap = cv2.VideoCapture('data/synthetic/escrime-4-3.avi')


########## Function ##########


def calculate_histogramHSV(image, tracked_area):
    x, y, w, h = tracked_area

    roi = image[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    channels = [0, 1, 2]
    hist_size = [30, 32, 32]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    v_ranges = [0, 256]

    roi_hist = cv2.calcHist([hsv_roi], channels, None, hist_size, h_ranges + s_ranges + v_ranges)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist.flatten()


def calculate_histogramRGB(image, tracked_area):
    x, y, w, h = tracked_area
    roi = image[y:y+h, x:x+w]

    channels = [0, 1, 2]
    hist_size = [32, 32, 32]
    ranges = [0, 256, 0, 256, 0, 256]

    roi_hist = cv2.calcHist([roi], channels, None, hist_size, ranges)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist.flatten()


def initialize_tracking(video_capture,type = 'RGB'):
    _,frame=video_capture.read()

    roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    if type == 'RGB':
        roi_hist = calculate_histogramRGB(frame, roi)
    elif type == 'HSV':
        roi_hist = calculate_histogramHSV(frame, roi)
    return roi_hist, roi


def initialize_particles(roi, num_particles):
    x, y, w, h = roi
    center=(x+w//2, y+h//2)

    particles=np.array([np.random.normal(center, 1) for _ in range(num_particles)])
    weights=np.ones(num_particles)/num_particles

    return particles, weights


def predict_particles(particles,sigma):
    noise = np.random.randn(*particles.shape) * sigma
    particles += noise
    return particles


def weights_update(particles,frame,hist_ref,roi_size, type = 'RGB', lamda=3):
    weights=np.zeros(particles.shape[0])

    for i, particle in enumerate(particles):
        x, y = particle
        w,h = roi_size
        roi= (int(x-w//2), int(y-h//2), w, h)

        if type == 'RGB':
            roi_hist=calculate_histogramRGB(frame,roi)
        elif type == 'HSV':
            roi_hist=calculate_histogramHSV(frame,roi)

        distance=cv2.compareHist(hist_ref,roi_hist,cv2.HISTCMP_BHATTACHARYYA)
        weights[i]=np.exp(-lamda*(distance**2))
    weights /= np.sum(weights)

    return weights


def resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)

    indexes = np.zeros(N, dtype=int)
    i, j = 0, 0
    u = np.random.rand() / N
    for i in range(N):
        u_i = u + i / N
        while u_i > cumulative_sum[j]:
            j += 1
        indexes[i] = j

    resampled_particles = particles[indexes]
    return resampled_particles


def estimate_position(particles, weights):
    return np.average(particles, weights=weights, axis=0)


def visualize_tracking(frame, particles, pos_estimate, roi_size):
    # Draw the rectangle around the estimated position
    x, y = pos_estimate
    w, h = roi_size
    top_left = (int(x - w / 2), int(y - h / 2))
    bottom_right = (int(x + w / 2), int(y + h / 2))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Draw each particle as a small dot
    for particle in particles:
        px, py = int(particle[0]), int(particle[1])
        cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

    return frame


########## Run ##########


    ########## Parameters ##########

num_particles = 100
sigma = np.array([4,4])
lamda = 3
type = 'RGB'            # 'RGB' or 'HSV'


    ########## While loop ##########

roi_hist,roi = initialize_tracking(cap,type)
particles, weights = initialize_particles(roi, num_particles)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    particles=predict_particles(particles,sigma)
    weights=weights_update(particles,frame,roi_hist,roi[2:], type, lamda)
    particles = resample(particles,weights)
    pos_estimate=estimate_position(particles,weights)
    visualize_tracking(frame,particles,pos_estimate,roi[2:])

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()