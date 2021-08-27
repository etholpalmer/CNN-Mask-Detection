from tensorflow.python.keras.backend import shape
import numpy as np
import os
import cv2

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

class Face_Detect:
    def __init__(self
            , file_name
            , confidence=0.5
            , proto="face_detector/deploy.prototxt"
            , wghts="face_detector/res10_300x300_ssd_iter_140000.caffemodel") -> None:

        self.file_name            = file_name if os.path.exists(file_name) else None
        self.image                = cv2.imread(self.file_name)
        self.results              = {}        
        (self.height, self.width) = self.image.shape[:2]
        print(f"The file ({self.file_name}) exists => {os.path.exists(self.file_name)}")

        @classmethod
        def get_face(X_begin, Y_begin, X_end, Y_end):
            if self.image is not None:
                face = self.image[Y_begin:Y_end, X_begin:X_end]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(src=face,dsize=(224,224))
                face = img_to_array(face)
                return np.expand_dims(face, axis=0)
            else:
                return None
        @classmethod
        def get_measurements(poss_faces):

            outline = poss_faces[0,0,x, 3:7] * np.array([self.width,self.height,self.width,self.height])
            X_begin, Y_begin, X_end, Y_end = outline.astype("int")
            
            return max(0,X_begin), max(0, Y_begin), min(self.width -1, X_end), min(self.height-1, Y_end)

        # Face Detector
        p = os.path.altsep.join(['.',proto])
        w = os.path.altsep.join(['.',wghts])
        print(os.path.exists(p),os.path.exists(w))
        net_mdl = cv2.dnn.readNet(p,w)
        # self.image = cv2.imread(filename=self.file_name)
        fixed_param = (104.0, 177.0, 123.0)
        img_blob = cv2.dnn.blobFromImage(self.image,1,(300,300), fixed_param)
        net_mdl.setInput(img_blob)

        possible_faces = net_mdl.forward()

        faces_info = []
        # for each face detected with a confidence level
        # above the threshold keep a reference of it.
        for x in range(possible_faces.shape[2]):
            conf = possible_faces[0,0,x,2]
            if conf > confidence:
                x1,y1,x2,y2 = get_measurements(self, poss_faces=possible_faces)
                faces_info.append((get_face(self, x1,y1,x2,y2),conf))

        self.results[file_name]=faces_info
    def __del__(self):
        print("Exiting Face Detect Class.")
        del self.image

if __name__ == "__main__":
    file1 = Face_Detect("./dataset/with_mask/100-with-mask.jpg")

    for fi in file1.results[file1.file_name]:
        print(fi[0].shape)