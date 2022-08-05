import pickle
import os
import dlib
import numpy as np
import cv2
    
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Haar
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat") #CNN

bd = []
rootdir = os.getcwd()+'\lfw'

i = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        
        img = cv2.imread(os.path.join(subdir, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #detection
        img_detection = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        #for k, d in enumerate(img_detection):
        for x,y,w,h in img_detection:
            #landmarks
            detected = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
            img_shape = sp(gray, detected)
            
            #alignment
            img_aligned = dlib.get_face_chip(img, img_shape)
            
            #representacion cnn
            img_representation = facerec.compute_face_descriptor(img_aligned)
            img_representation = np.array(img_representation)
           
            bd.append(img_representation)

pickle_file = open('representaciones_imgs_haar.txt', 'w+b')
pickle.dump(bd, pickle_file)
pickle_file.close()