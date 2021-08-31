from face_detect import Face_Detect
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from numpy import ndarray

class Mask_Detect:
    def __init__(self
            , mask_mdl:str="mask_detector.model"
    ) -> None:
        (self.mask_conf, self.no_mask_conf) = (0.0, 0.0)
        self.face_mask_mdl = load_model(mask_mdl)
    def __del__(self):
        print("Exiting Mask Detect Class.")
        del self.face_mask_mdl

    def Mask_Check(self, face:ndarray):
        if face is not None:
            (mask, no_mask) = self.face_mask_mdl.predict(face)[0]
            return ("Mask" if mask > no_mask else "Not Masked"), max(mask,no_mask)
        else:
            return "No Data",1.0

if __name__ == "__main__":
    import face_detect as fd
    file_info = fd.Face_Detect("./Dataset/test_photos/with_mask/0-with-mask.jpg")
    
    f = Mask_Detect()
    print(f.Mask_Check(file_info.results[file_info.file_name][0][0]))
