import cv2
import os

class Face_Detector:
    def __init__(self) -> None:
            # , proto="face_detector/deploy.prototxt"
            # , wghts="face_detector/res10_300x300_ssd_iter_140000.caffemodel") -> None:
        # # Configure Face Dector Model
        # p = os.path.altsep.join(['.',proto])
        # w = os.path.altsep.join(['.',wghts])
        # if (os.path.exists(p) and os.path.exists(w)):
        #     self.net_mdl = cv2.dnn.readNet(p,w)
        # else:
        #     self.net_mdl = None
        pass

    @staticmethod
    def Get_Mdl(proto="face_detector/deploy.prototxt"
            , wghts="face_detector/res10_300x300_ssd_iter_140000.caffemodel"):
        # Configure Face Dector Model
        p = os.path.altsep.join(['.',proto])
        w = os.path.altsep.join(['.',wghts])
        if (os.path.exists(p) and os.path.exists(w)):
            return cv2.dnn.readNet(p,w)
        else:
            return None

    def __del__(self):
        print("Face_Detector Exited.")

if __name__ == "__main__":
    mdl = Face_Detector().Get_Mdl()
    img = cv2.imread(".Dataset/test_photos/with_mask/100-with-mask.jpg")
    fixed_param = (104.0, 177.0, 123.0)
    img_blob = cv2.dnn.blobFromImage(img,1,(300,300), fixed_param)
    mdl.setInput(img_blob)
    print(mdl.forward())
