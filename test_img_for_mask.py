# Import necessary modules
import timeit
start_module_load = timeit.default_timer()
import  mask_detect as md, face_detect as fd, face_detector as fdd
print(f"Took {timeit.default_timer()-start_module_load} to load ... the modules")

# Import argparse to write user-friedly command-line interfaces

import argparse
# import cv2
import os

from numpy import record

# Construct the argument parser and parse the arguments 
# The path to dataset of testing images (with masks) on disk
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	default="./dataset/with_mask",
	help="path to input images directory")
# args = vars(ap.parse_args(args=["-d=dataset/without_mask"]))
args = vars(ap.parse_args(args=["-d=examples"]))
# args = vars(ap.parse_args(args=["-d=dataset/with_mask"]))

# Get all the files from the args["dir"] folder
filenames = [
                (os.path.altsep.join([args["dir"],x])) for x in os.listdir(args["dir"]) 
                if os.path.splitext(x)[1].lower()=='.jpg'
                or os.path.splitext(x)[1].lower()=='.png'
            ]

chkMask = md.Mask_Detect(mask_mdl="./models/mask_detector-epoch-1.model")

# Display results/figures for all images if there is face detected
records = {}
for file in filenames:
    fi = fd.Face_Detect(file_name=file)
    if fi.There_is_a_face():
        records[fi.file_name] = chkMask.Mask_Check(
                                    fi.results[fi.file_name][0][0]
                                )
    del fi
for rec in records:
    print(f"File:{rec} has {records[rec]}")
print(records)