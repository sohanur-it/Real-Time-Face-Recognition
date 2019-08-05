import os
import cv2
import pickle
import numpy as np
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(BASE_DIR)  /home/Documents/OpenCV

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

image_dir = os.path.join(BASE_DIR, "images")

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)##sohan
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(id_)

            pil_image = Image.open(path).convert(
                "L")  # Convert into gray scale
            #size = (550, 550)
            # final_image = pil_image.resize(size, Image.ANTIALIAS)
            # image_array = np.array(final_image, "uint8")
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, h, w) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open("face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)


print(x_train)
print(np.array(y_labels))
print(len(x_train))
print(len(y_labels))


recognizer.train(x_train, np.array(y_labels))
recognizer.save("facetrainner.yml")
