from tensorflow.python.keras.backend import shape
import numpy as np
import os
import cv2

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import face_detector as fdd

class Face_Detect:
    def __init__(self
            , file_name
            , confidence=0.5
            , net_mdl=fdd.Face_Detector.Get_Mdl()) -> None:

        success         = False
        self.results    = {}

        self.file_name                = file_name if os.path.exists(file_name) else None
        if self.file_name is not None:
            self.image                = cv2.imread(self.file_name)
            (self.height, self.width) = self.image.shape[:2]
            self.net_mdl              = net_mdl

            print(f"The file ({self.file_name}) exists => {os.path.exists(self.file_name)}")
            
            def get_face(X_begin, Y_begin, X_end, Y_end):
                try:
                    if self.image is not None:
                        face = self.image[Y_begin:Y_end, X_begin:X_end]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = cv2.resize(src=face,dsize=(224,224))
                        face = img_to_array(face)

                        successful_process = True

                        rVal = (successful_process, np.expand_dims(face, axis=0))
                    else:
                        rVal = (False, None)
                except Exception as excn:
                    print("This is an error:", excn.args)
                    rVal = (False, None)
                else:
                    pass

                return rVal

            def get_measurements(poss_faces):
                outline = poss_faces[0,0,x, 3:7] * np.array([self.width,self.height,self.width,self.height])
                X_begin, Y_begin, X_end, Y_end = outline.astype("int")

                return max(0,X_begin), max(0, Y_begin), min(self.width -1, X_end), min(self.height-1, Y_end)

            # self.image = cv2.imread(filename=self.file_name)
            fixed_param = (104.0, 177.0, 123.0)
            img_blob = cv2.dnn.blobFromImage(self.image,1,(300,300), fixed_param)
            self.net_mdl.setInput(img_blob)

            possible_faces = self.net_mdl.forward()

            faces_info = []
            # for each face detected with a confidence level
            # above the threshold keep a reference of it.
            for x in range(possible_faces.shape[2]):
                conf = possible_faces[0,0,x,2]
                if conf > confidence:
                    x1,y1,x2,y2 = get_measurements(poss_faces=possible_faces)
                    success, f = get_face(x1,y1,x2,y2)
                    if success and f is not None:
                        faces_info.append((f,conf,(x1,y1,x2,y2)))
                    else:
                        break
            if success:
                self.results[file_name]=faces_info
        else:
            self.results[file_name] = [(None,0.0,(0,0,0,0))]

    def There_is_a_face(self):
        return (self.file_name is not None and len(self.results)>0)
    def Number_of_possible_faces_detected(self):
        if (self.file_name is not None):
            return len(self.results)
        else:
            return 0
    def Get_Original_Image(self):
        if (self.file_name is not None):
            return self.image
        else:
            return None

    def Get_File_Name(self):
        return self.file_name

    def __del__(self):
        print("Exiting Face Detect Class.")
        del self.image

if __name__ == "__main__":
    file1 = Face_Detect("./dataset/with_mask/50-with-mask.jpg")
    colour = (0,255,0)
    if file1.There_is_a_face():
        for img,conf,(x1,y1,x2,y2) in file1.results[file1.file_name]:
            cv2.rectangle(file1.Get_Original_Image()
                    , (x1,y1), (x2,y2)
                    , color=colour
                    , thickness=2)
            print(img.shape)

        cv2.imshow(file1.file_name, file1.Get_Original_Image())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(file1.There_is_a_face())
        print("Could not find any faces.  File may be missing.")