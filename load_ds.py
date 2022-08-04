import pickle
from imutils import paths
import os
import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector() #HOG
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat") #CNN

bd = []
rootdir = os.getcwd()+'\lfw'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        img = dlib.load_rgb_image(os.path.join(subdir, file))
        
        #detection
        img_detection = detector(img, 1)
        
        #for k, d in enumerate(img_detection):
        for detected in img_detection:
            #landmarks
            img_shape = sp(img, detected)
            
            #alignment
            img_aligned = dlib.get_face_chip(img, img_shape)
            
            #representacion cnn
            img_representation = facerec.compute_face_descriptor(img_aligned)
            img_representation = np.array(img_representation)
           
            bd.append(img_representation)

pickle_file = open('representaciones_imgs.txt', 'w+b')
pickle.dump(bd, pickle_file)
pickle_file.close()