import cv2
import os

# Charger la vidéo
video_path = "data/synthetic/escrime-4-3.avi"  # Assure-toi que la vidéo est téléchargée dans Colab ou spécifie le chemin complet

video_capture = cv2.VideoCapture(video_path)

# Créer un dossier pour enregistrer les frames
output_folder = 'frames'
os.makedirs(output_folder, exist_ok=True)

# Sauvegarder les frames en images
frame_count = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Lire une frame

    if not ret:
        break

    frame_path = f'{output_folder}/frame_{frame_count}.jpg'
    cv2.imwrite(frame_path, frame)  # Enregistrer la frame en tant qu'image

    frame_count += 1

video_capture.release()

import matplotlib.pyplot as plt

# Parcourir les images enregistrées et les afficher
for i in range(frame_count):
    frame_path = f'frames/frame_{i}.jpg'
    frame = cv2.imread(frame_path)

    # Afficher l'image
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()