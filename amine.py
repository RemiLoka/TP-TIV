import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('data/synthetic/escrime-4-3.avi')


def calculate_histogram(image, tracked_area):
    x, y, w, h = tracked_area
    roi = image[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist



def initialize_tracking(video_capture):
  _,frame=video_capture.read()

  roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow("Select Object")

  roi_hist = calculate_histogram(frame, roi)
  return roi_hist, roi



def initialize_particles(roi, num_particles):
    x, y, w, h = roi
    center=(x+w//2, y+h//2)

    paricles=np.array([np.random.normal(center, 20) for _ in range(num_particles)])
    weights=np.ones(num_particles)/num_particles

    return paricles, weights



def predict_particles(particles,sigma):
  noise = np.random.randn(particles.shape) * sigma
  particles += noise
  return particles
   


roi, roi_hist = initialize_tracking(cap)

plt.bar(range(len(roi_hist)), roi_hist, width=1)
plt.show()
    
