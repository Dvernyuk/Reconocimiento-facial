import argparse
import pickle
import numpy as np
import dlib
import os
import cv2

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
    
def haar(img, detected, gray):
    #for x,y,w,h in img_detection:
    x,y,w,h = detected
    #landmarks
    detected = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
    img_shape = sp(gray, detected)
    
    #alignment
    img_aligned = dlib.get_face_chip(img, img_shape)
    
    #representacion cnn
    img_representation = facerec.compute_face_descriptor(img_aligned)
    img_representation = np.array(img_representation)
    return img_representation
    

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Haar
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat") #CNN

bd = []
pickle_file = open('representaciones_imgs_hog.txt', 'r+b')
bd = pickle.load(pickle_file)
pickle_file.close()
rootdir = os.getcwd()+'\lfw'

score = 0
samples = 0
threshold = 0.5

for subdir, dirs, files in os.walk(rootdir):
    end = False
    tail = files
    while tail != [] and not end:
        head, *tail = tail
        
        img1 = cv2.imread(os.path.join(subdir, head))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        #detection
        img_detection1 = detector.detectMultiScale(gray1, scaleFactor=1.2, minNeighbors=5)
        
        # Solo nos enfocamos en las fotos que detectamos una o ninguna cara
        if not(len(img_detection1) > 1):
            if len(img_detection1) == 0: #La no-deteccion se penaliza
                samples += 0
            elif len(files) == 1: #Una deteccion simple no se premia
                samples += 0
                score += 0
            else:
                end = True
                
                img_representation1 = haar(img1, img_detection1[0], gray1)
                    
                correct = 0
                for file in files:
                    if head != file:
                        img2 = cv2.imread(os.path.join(subdir, file))
                        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        
                        #detection
                        img_detection2 = detector.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=5)
                        
                        best = 1
                        for detected in img_detection2:
                            img_representation2 = haar(img2, detected, gray2)
                            aux = findEuclideanDistance(img_representation1, img_representation2)
                            if best > aux:
                                best = aux
                        
                        if(best < threshold):
                            correct += 1
                        
                score += correct / (len(files)-1)
                samples += 1
                  
file = open('facial_recognition_test_haar_v2_result.txt', 'w')
file.write("Ratio de acierto: "+str(score/samples*100)+"%")
file.close()