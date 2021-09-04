# Import necessary modules
import  mask_detect as md, face_detect as fd, face_detector as fdd
# Import argparse to write user-friedly command-line interfaces
import argparse

# import cv2

import os

from numpy import record

# Construct the argument parser and parse the arguments 
 # The path to dataset of testing images (with masks) on disk
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	default="./Dataset/test_photos/with_mask",
	help="path to input images directory")
args = vars(ap.parse_args(args=["-d=Dataset/test_photos/with_mask"]))

# Get all the files from the args["dir"] folder
filenames = [
                (os.path.altsep.join([args["dir"],x])) for x in os.listdir(args["dir"]) 
                if os.path.splitext(x)[1].lower()=='.jpg'
                or os.path.splitext(x)[1].lower()=='.png'
            ]

chkMask = md.Mask_Detect()

# Display results/figures for all images if there is face detected
records = {}
for file in filenames:
    fi = fd.Face_Detect(file_name=file)
    if fi is not None and len(fi.results)>0:
        records[fi.file_name] = chkMask.Mask_Check(
                                    fi.results[fi.file_name][0][0]
                                )
    del fi

print(records)