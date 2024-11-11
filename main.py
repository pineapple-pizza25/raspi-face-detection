import cv2
import time
import requests
import io
from picamera2 import Picamera2

api_url = 'https://facial-recognition-api.calmwave-03f9df68.southafricanorth.azurecontainerapps.io/facialrecognition'

clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

face_detected = False


while True:
    frame = picam2.capture_array()
    
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors = 10,
        minSize = (30,30) ,
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        #time.sleep(1)
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x,y), (x+width, y+width), (255, 255, 0), 2)

        #result, image = picam2.capture_array()
        picam2.capture_file("face.jpg")

        #cv2.imshow("image", image)
        cv2.waitKey(1) 
        time.sleep(6)
        cv2.destroyWindow("image")

    
        try:
            with open("face.jpg", "rb") as file:
                response = requests.post(api_url, files={"image": file})
                print('Status Code:', response.status_code)
                print('Response Data:', response.json())
                
        except requests.exceptions.RequestException as e:
            print('Error:', e)
            #return None
        


        print('Success!')


        faces = []        
    else:
        face_detected = False

    if cv2.waitKey(1) == ord("q"):
        break


cv2.destroyAllWindows()