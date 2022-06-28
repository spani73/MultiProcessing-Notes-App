# Importing the libraries
def imageDetector():
    from PIL import Image
    from keras.applications.vgg16 import preprocess_input
    import base64
    from io import BytesIO
    import json
    import random
    import cv2
    from keras.models import load_model
    import numpy as np
    import time
    import multiprocessing
    import logging
    from multiprocessing.context import Process
    import saver

    from keras.preprocessing import image
    model = load_model('facefeatures_new_model.h5')

    # Loading the cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    worker = None


    def face_extractor(img):
        # Function detects faces and returns the cropped face
        # If no face detected, it returns the input image
        
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        if faces == ():
            return None
        
        # Crop all faces found
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face

    # Doing some Face Recognition with the webcam

    while True:
        video_capture = cv2.VideoCapture(0)
        _, frame = video_capture.read()
        
        face=face_extractor(frame)
        ##print(face)
        name="No Faces"
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
            #Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
                        #Our keras model used a 4D tensor, (images x height x width x channel)
                        #So changing dimension 128x128x3 into 1x128x128x3 
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            print(pred)
                        
            
            if(pred[0][0]>0.5):
                name='Subhankar'
            else:
                name="No Faces"
        ##cv2.imshow('Video', frame)
        video_capture.release()
        print(multiprocessing.current_process().name + "in test")
        if name == "No Faces" and (worker == None or worker.is_alive() == False):
            print(name + " in 1st condition")
            worker = Process(target=saver.work,name='Worker')
            worker.start()
        elif name== "Subhankar" and (worker == None or worker.is_alive() == False):
            pass
        elif name == "No Faces" and worker.is_alive() == True:
            pass 
        else:
           worker_alive = worker.is_alive()
           print(name + "in last condition" + str(worker_alive)) 
           worker.terminate()   
        
        time.sleep(5)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("e"):
            break


    worker.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    imageDetector()



       

