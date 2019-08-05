import cv2
import time
import pickle
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("facetrainner.yml")

labels = {}
with open("face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    print(og_labels)
    labels = {v: k for k, v in og_labels.items()}

print(labels)


video = cv2.VideoCapture(0)


while(True):
    check, frame = video.read()
    # print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]  # (y_cord_start,y_cord_end)
        roi_color = frame[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        # print(id_)

        print(labels[id_])
        print(conf)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255, 255, 255)
        stroke = 3
        cv2.putText(frame, name, (x, y), font,
                    3, color, stroke, cv2.LINE_AA)
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        # Frame Style
        color = (255, 0, 0)  # BGR 0-255
        stroke = 3
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow("capturing", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
