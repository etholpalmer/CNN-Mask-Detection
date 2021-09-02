# USAGE
# python train_mask_detector.py --dataset dataset

# Import the necessary packages
# Import tensorflow.keras for data augmentation, fine-tune model, building new fully-connected head, pre-processing and loading image data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
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
	default="mask_detector.model",
	help="path to output face mask detector model")
#args = vars(ap.parse_args())
args = {}
args["dataset"] = "./dataset"
args["plot"]    = "plot.png"

# Initialize the rate for initial learning rate, the number of epochs to train for, and the size for batch
 # Start range test from lower learning rate = 1e-4
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Grab all image paths in dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# Initialize the list of data and class images 
data = []
labels = []

# Loop over the image paths (loading and pre-processing images)
for imagePath in imagePaths:
	# Extract the class label from the file's name
	label = imagePath.split(os.path.sep)[-2]
    
	# Load the image and resize the image to 224x224 pixels
	image = load_img(imagePath, target_size=(224, 224))
	# Convert the image to array format
	image = img_to_array(image)
	# Scale the pixel intensities of the image to the range [-1, 1]
	image = preprocess_input(image)

	# Update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels (each element of the labels array consists of an array in which only one index is “hot” (i.e., 1))
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data into 80% data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Construct the training image generator for data augmentation (Replacing the original batch with the new, randomly transformed batch to improve the generalization )
aug = ImageDataGenerator(
	rotation_range=20,I
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Apply fine-tuning strategy to neural network
# Step 1: Establish baseline model by loading the MobileNetV2 with pre-trained ImageNet weights (without the fully-connected layer heads - Classification)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Step 2: Construct a new set of fully connected layer heads
headModel = baseModel.output
# Reduce the size of the inputs
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# Expands into a one-dimensional vector
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
# Reduce over fitting
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Replace the heads with the new fully connected laryers (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Step 3: Freeze the base layers of the network
# Loop over all layers in the base model and freeze them to prevent the weights in a given layer from being updated during training
for layer in baseModel.layers:
	layer.trainable = False

# Compile the model with Adam optimizer (adjust learning rate by using learning rate schedules - decrease the learning rate 
# to allow the network to take less steps while maintain reasonable accuracy) and binary cross-entropy (for two classes)
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	# Perform validation to learn the accuracy and loss of the model
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Perform predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Keep the highest probability class label indices for each image in the testing set
predIdxs = np.argmax(predIdxs, axis=1)

# Refine the format for the classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Save the model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

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