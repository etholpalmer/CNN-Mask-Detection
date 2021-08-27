import  mask_detect as md, face_detect as fd

import argparse
import cv2
import os

from numpy import record

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	default="./dataset/with_mask",
	help="path to input images directory")
args = vars(ap.parse_args(args=["-d=dataset/with_mask"]))

# Get all the files from the args["dir"] folder
filenames = [
                (os.path.altsep.join([args["dir"],x])) for x in os.listdir(args["dir"]) 
                if os.path.splitext(x)[1].lower()=='.jpg'
                or os.path.splitext(x)[1].lower()=='.png'
            ]

chkMask = md.Mask_Detect()

records = {}
for file in filenames:
    fi = fd.Face_Detect(file_name=file)
    records[fi.file_name] = chkMask.Mask_Check(
                                fi.results[fi.file_name][0][0]
                            )

print(records)