import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
rec = cv2.face.createLBPHFaceRecognizer();
rec.load("Recognizer/trainingData.yml")
id=0
name = ""
font = cv2.FONT_HERSHEY_SIMPLEX

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


while(True):
    ret,img=cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2)
        id=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            name="Rasmus Toivanen"
        elif(id==2):
            name="Elon Musk"
        cv2.putText(img,str(name),(x,y+h+30),font,0.8,(0,255,0)),2;
        out.write(img)
    cv2.imshow('Face',img);
    if(cv2.waitKey(1)==ord('q')):
       break;
      
cam.release()
out.release()
cv2.destroyAllWindows()
    
