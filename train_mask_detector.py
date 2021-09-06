# USAGE
# python train_mask_detector.py --dataset dataset

# Import the necessary packages
# Import tensorflow.keras for data augmentation, fine-tune model, building new fully-connected head, pre-processing and loading image data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical, plot_model

# Import from sklearn for binarizing class labels, segmenting dataset, and printing classification report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import imutils paths to find and list images in the dataset
from imutils import paths

# Import matplotlib to plot training curves
import matplotlib.pyplot as plt

# Import numpy for numberical processing
import numpy as np

# Import argparse to write user-friedly command-line interfaces
import argparse

# Import os to build file/directory paths directly 
import os

# Construct the argument parser and parse the arguments 
 # 1. The path to dataset of images on disk
 # 2. The path to the output training plot
 # 3. The path to the output mask detector model
 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="./Models/original_mask_detector.model",
	help="path to output face mask detector model")
ap.add_argument("-e", "--epochs",     type=int, 	default=20, 			help="The number of epochs to generate")
ap.add_argument("-b", "--batchsize",  type=int, 	default=32, 			help="Batch Size")
ap.add_argument("-bm", "--basemodel", type=str, 	default="MobileNetV2", 	help="Base Model")
ap.add_argument("-lr", "--learnrate", type=float, 	default=1e-4, 			help="The Learning Rate")
args = vars(ap.parse_args())
#args = vars(ap.parse_args(args=["-d=./dataset"]))
#args = vars(ap.parse_args(args=["-d=examples"]))

# args = {}
# args["dataset"] = "./dataset"
mdl_details 	= f"{args['epochs']}_batch-{args['batchsize']}_basemodel-{args['basemodel']}_learnrate-{args['learnrate']}"
args["plot"]    = f"./Results/plot_epoch-{mdl_details}.png"
args["model"]	= f"./models/mask_detector_epoch-{mdl_details}.model"

# print(mdl_details)
# print(args["epochs"])
# print(args)
# exit(0)

# Initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = args['learnrate']	# 1e-4
EPOCHS = args['epochs'] # 20
BS = args['batchsize']	# 32

# Grab the list of images in our dataset directory, then initialize the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Loop over the image paths
for imagePath in imagePaths:
	# Extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# Load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# Update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.20, stratify=labels, random_state=42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

if args["basemodel"] == "MobileNetV2":
	# Load the MobileNetV2 network, ensuring the head FC layer sets are left off
	baseModel = MobileNetV2(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))
else:
	baseModel = MobileNetV2(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(x=testX, batch_size=BS)

# For each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Show a nicely formatted classification report
class_report=str(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
with open(f"./Results/class_report_{mdl_details}.txt","w") as save:
	save.write(class_report)
print(class_report)

# Save the model
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# Save the model Summary (text)
mdl_summary = str(model.summary())
with open(f"./Results/model_summary_{mdl_details}.txt","w") as save:
	save.write(mdl_summary)
print(mdl_summary)

# Save the model structure
plot_model(model, to_file=f"./Results/model_struct_{mdl_details}.jpg")

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])