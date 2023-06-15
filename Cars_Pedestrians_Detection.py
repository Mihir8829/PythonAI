import cv2

# all the trained classifier files
car_classifier_file = 'cars.xml'
pedestrian_calssifier_file = 'pedestrian.xml'
twoWheeler_classifier_file = 'two_wheelers.xml'

# Have to capture before reading VIDEOS!
car_img = 'Media/cars.jpg'
video = cv2.VideoCapture('Media/pedestrian_video3.mp4')
pedestrian_img = 'Media/pedestrians.jpg'

# trained model classifier for cars and Pedestrains
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_calssifier_file)


while True:
    # gives all consecutive frames from video   ***read() gives two values -> boolean and frame value(array)***
    (read_successful, frame) = video.read()

    if read_successful:
        grey_scaledFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detects multiple cars AND pedestrians per frame
    cars = car_tracker.detectMultiScale(grey_scaledFrame)
    pedestrians = pedestrian_tracker.detectMultiScale(grey_scaledFrame)

    # loop over every frame to make rectangle over cars and pedestrians
    for x, y, w, h in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)

    cv2.imshow("Cars and Pedestrians", frame)
    key = cv2.waitKey(1)  # waits for user to input to end showing

    if key == 81 or key == 113:
        break
video.release()

print("code completed")


# Code for Images of Cars
'''
img = cv2.imread(car_img)  # Must read before showing image/video
blackWhite = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #converts img to black and white
cars = car_tracker.detectMultiScale(blackWhite)  #detects multiple cars in one frame/picture  --> collects frames around cars and stores it in list of mutliple cars 
for x, y, w, h in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
cv2.imshow("Cars", img)
cv2.waitKey()   #waits for user to input to end showing

'''
# Code for Images of Pedestrians
'''
img = cv2.imread(pedestrian_img)
peds = pedestrian_tracker.detectMultiScale(img)
for x,y,w,h in peds:
    cv2.rectangle(img,  (x, y), (x+w, y+h), (0, 255, 0), 1)
cv2.imshow("peds",img)
cv2.waitKey()
'''
