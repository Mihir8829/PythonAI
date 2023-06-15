import cv2

smile_detector = cv2.CascadeClassifier('smile.xml')
face_detector = cv2.CascadeClassifier('faces.xml')
webcam = cv2.VideoCapture(0)  # web cam feed
while True:
    succesfull_frame_read, frame = webcam.read()

    if not succesfull_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale)

    # runs for faces -> rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # gets the subFrame from the frame of the face
        the_face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(
            the_face, cv2.COLOR_BGR2GRAY)  # makes the subFrame gray

        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)
        # runs for smiles -> rectangle
        # for (X_, Y_, W_, H_) in smiles:
        #     cv2.rectangle(the_face, (X_, Y_), (X_+W_, Y_+H_), (0, 255, 0), 1)

        #puts label instead of rectangle around smile
        if len(smiles)>0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
    cv2.imshow('Why so serious', frame)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
