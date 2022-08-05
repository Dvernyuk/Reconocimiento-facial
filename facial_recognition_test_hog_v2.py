import argparse
import pickle
import numpy as np
import dlib
import os

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

detector = dlib.get_frontal_face_detector() #HOG
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
        
        img = dlib.load_rgb_image(os.path.join(subdir, head))

        #detection
        img_detection = detector(img, 1)
        
        # Solo nos enfocamos en las fotos que detectamos una o ninguna cara
        if not(len(img_detection) > 1):
            if len(img_detection) == 0: #La no deteccion se penaliza
                samples += 1
            elif len(files) == 1: #Una deteccion simple no se premia
                score += 0
                samples += 0
            else:
                end = True
                samples += 0
                
                detected = img_detection[0]
                #landmarks
                img_shape = sp(img, detected)
                
                #alignment
                img_aligned = dlib.get_face_chip(img, img_shape)
                
                #representacion cnn
                img_representation = facerec.compute_face_descriptor(img_aligned)
                img_representation = np.array(img_representation)
                    
                detected_bd = 0
                for face_bd in bd: #Recorremos por todo la bd para buscar todas las coincidencias
                    if findEuclideanDistance(img_representation, face_bd) < threshold:
                        detected_bd += 1
                
                real_matches = len(files) - 1
                if real_matches*2 > detected_bd:
                    score += (real_matches - abs(real_matches - detected_bd)) / real_matches
                  
file = open('facial_recognition_test_hog_result_v2.txt', 'w')
file.write("Ratio de acierto: "+str(score/samples*100)+"%")
file.close()