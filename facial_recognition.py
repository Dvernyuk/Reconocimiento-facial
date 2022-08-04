import argparse
import pickle
import numpy as np
import dlib

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector() #HOG
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat") #CNN

bd = []
pickle_file = file('representaciones_imgs.txt', 'r+b')
bd = pickle.load(pickle_file)

img = dlib.load_rgb_image(args["image"])
#detection
img_detection = detector(img, 1)

threshold = 0.6
for detected in img_detection:
    #landmarks
    img_shape = sp(img, detected)
    
    #alignment
    img_aligned = dlib.get_face_chip(img, img_shape)
    
    #representacion cnn
    img_representation = facerec.compute_face_descriptor(img_aligned)
    img_representation = np.array(img_representation)
    
    
    if any(findEuclideanDistance(img_representation, img_bd) < threshold for img_bd in bd)
        print('Detectado en la bd')
    else
        print('No detectado en la bd')