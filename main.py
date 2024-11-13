import cv2
import time
import requests
import io
from picamera2 import Picamera2


api_url = 'http://127.0.0.1:8000/facialrecognition'


clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

picam2.set_controls({"ExposureTime": 30000}) 

face_detected = False


while True:
    frame = picam2.capture_array()
    
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors = 5,
        minSize = (30,30) ,
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        #time.sleep(1)
        print("I picked up a face!!! You better be in the system")
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x,y), (x+width, y+width), (255, 255, 0), 2)

        print("Taking picture")
        picam2.capture_file("face.jpg")
        print("picture taken!!!!!")

        #cv2.imshow("image", image)
        cv2.waitKey(1) 
        time.sleep(6)

    
        try:
            print("I am going to upload this picture now")

            with open("face.jpg", "rb") as file:
                response = requests.post(api_url, files={"file": file})
                print('Status Code:', response.status_code)
                print('Response Data:', response.json())

                if response.status_code == 200:
                    print('Success!')
                
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