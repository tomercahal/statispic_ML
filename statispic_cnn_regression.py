# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.optimizers import Adam
from keras.callbacks.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from datasets import *
from models import *
import numpy as np
import argparse
import locale
import sys
import os


def train_model(X,Y):
    model = create_cnn(400, 400, 3, regress=True)
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    model.summary() # display a description of the model
    # filepath = "D:/Statispic2/try-best-weights.hdf5" #Each time overriding this file if there there are better weights
    # checkpoint = ModelCheckpoint(
    #     filepath, monitor='loss',
    #     verbose=1,
    #     save_best_only=True,
    #     mode='min'
    # )
    # callbacks_list = [checkpoint]

    #model.fit(X, Y, epochs=70, batch_size=8, callbacks=callbacks_list)
    model.fit(X, Y, epochs=70, batch_size=8)
    model.save("D:/statispic2/final-statispic_model.hdf5")

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description="Enter the dir where the images are saved")
parser.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(parser.parse_args()) #turning into a python dictionary
print("The args given are:")
print(args)

# construct the path to the input .txt file that contains information
# on each image in the dataset and then load the dataset
print("[INFO] Loading images attributes...")
inputPath = os.path.sep.join([args["dataset"], "statispicData.txt"])
df = load_pictures_attributes(inputPath) #From the datasets file

# load the images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading doggie images!!")
images = load_house_images(df, args["dataset"]) #From the datasets file
images = images / 255.0

split = train_test_split(df, images, test_size=0.00045, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
print(len(trainAttrX), len(testAttrX), len(trainImagesX), len(testImagesX))

# find the largest ratio in the training set and use it to
# scale the rest to the range [0, 1] (will lead to better
# training and convergence)
maxSuccessRatio = trainAttrX["success_ratio"].max()
trainY = trainAttrX["success_ratio"] / maxSuccessRatio

train_model(trainImagesX, trainY)