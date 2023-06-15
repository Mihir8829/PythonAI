import cv2
trained_face_data = cv2.CascadeClassifier('faces.xml')
img = cv2.imread('Media/front_faces.webp')
webcam = cv2.VideoCapture(0)
print("Welcome to My Face Detection APP!😁")

# Code for Images
'''
face_coordinates = trained_face_data.detectMultiScale(img)

for x, y, w, h in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)

cv2.imshow("Images", img)
cv2.waitKey()
'''

while True:
    successful_frame, frame = webcam.read()
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    for x, y, w, h in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("RDJ here", frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()

print("Program Completed")
