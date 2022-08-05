import argparse
import pickle
import numpy as np
import dlib
import cv2

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Haar
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat") #CNN

bd = []
pickle_file = open('representaciones_imgs_haar.txt', 'r+b')
bd = pickle.load(pickle_file)
pickle_file.close()

img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#detection
img_detection = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

threshold = 0.5
for x,y,w,h in img_detection:
    #landmarks
    detected = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
    img_shape = sp(gray, detected)
    
    #alignment
    img_aligned = dlib.get_face_chip(img, img_shape)
    
    #representacion cnn
    img_representation = facerec.compute_face_descriptor(img_aligned)
    img_representation = np.array(img_representation)
    
    
    if any(findEuclideanDistance(img_representation, face_bd) < threshold for face_bd in bd):
        print('Detectado en la bd')
    else:
        print('No detectado en la bd')