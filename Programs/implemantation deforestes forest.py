from ctypes import resize

from pandas import array
from ModeleUnetSegmentationBinaire import build_unet

import tensorflow as tf

from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import glob

import torch 
import torch.nn as nn 
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

import rasterio
from rasterio.plot import show

from sklearn.model_selection import train_test_split

import skimage.io
import pandas as pd
import random

#image_directory = 'C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/database landsat 8 ariquemes/'

SIZE = 256
num_images = 100

# Data Paths
ROOT_IMAGE_DIR = 'C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/images\\'
ROOT_MASK_DIR = 'C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/masks\\'
METADATA_CSV_PATH = 'C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/meta_data.csv'
#images = []

# images = Image.open('C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/images/*.jpg')
# image = Image.open('C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/images/855_sat_01.jpg')
# image = np.array(images[0])

# Read the images
arrayed_images = []
arrayed_masks = []
#image_name = glob.glob("C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/images/*.jpg")


image_names = glob.glob("C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/images/*.jpg")
image_names_subset = image_names[0:num_images]

# image = Image.open(image_names[0])
# image = np.array(image)
for i,img in enumerate(image_names_subset):
    img = Image.open(img)
    array_img = np.array(img)
    arrayed_images.append(array_img)

mask_names = glob.glob("C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/masks/*.jpg")
mask_names_subset = mask_names[0:num_images]

# image = Image.open(image_names[0])
# image = np.array(image)
for i,img in enumerate(mask_names_subset):
    img = Image.open(img)
    array_img = np.array(img)
    arrayed_masks.append(array_img)

# plt.figure(0)
# plt.imshow(arrayed_images[20])
# plt.show()

#print(arrayed_images[20])
#image = skimage.io.imread('C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/images')
#image = np.array(image)
# for i,img in enumerate(image_names_subset):
#     new_img = Image.fromarray(img).resize((256, 256))
#     new_img = np.array(new_img)
#     resized_image_dataset.append(new_img)

random.shuffle(arrayed_images)
random.shuffle(arrayed_masks)

import csv
# Chargement des correspondances depuis le fichier CSV
correspondances = {}
with open('C:/Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/DATA/Forest Segmented/meta_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(reader):
        correspondances[row[0]] = row[1]
        if i == 100 :
            break;

# Création de deux nouvelles listes triées en fonction des correspondances
sorted_images = []
sorted_masks = []
for image in image_names_subset:
    image = image.split("\\")[-1]
    sorted_images.append(image)
    sorted_masks.append(correspondances[image])

arrayed_sorted_images = []
for i,img in enumerate(sorted_images):
    img = ROOT_IMAGE_DIR + img
    img = Image.open(img)
    array_img = np.array(img)
    arrayed_sorted_images.append(array_img)

arrayed_sorted_masks = []
for i,img in enumerate(sorted_masks):
    img = ROOT_MASK_DIR + img
    img = Image.open(img)
    array_img = np.array(img)
    arrayed_sorted_masks.append(array_img)

    
# plt.figure(0)
# plt.imshow(arrayed_images[25])
# plt.figure(1)
# plt.imshow(arrayed_masks[25])
# plt.show()

# # Load CSV File
# metadata = pd.read_csv(METADATA_CSV_PATH)

# # Quick look
# print(metadata.head())


# #dataset = image_subset
dataset = np.array(arrayed_sorted_images)
maskset = np.array(arrayed_sorted_masks)

# dataset2 = np.array(arrayed_images)
# maskset2 = np.array(arrayed_images)

# #dataset.Permute(1,2,3,0)
print('rr')


print("Image data shape is: ", dataset.shape)
print("Image data shape is: ", maskset.shape)
print("Max pixel value in image is: ", dataset.max())



# X_train.append(resized_image)
# X_test = resized_image
# y_train = resized_image
# y_test = resized_image

X_train, X_test, y_train, y_test = train_test_split(dataset, maskset, test_size = 0.2, random_state = 42)
#plt.figure(0)
# plt.imshow(X_train[0])
#plt.figure(1)
#plt.imshow(X_test[0])
#plt.figure(0)
#plt.imshow(y_test)
#plt.show()

# --- MODELE UNET, TRAINING DAY --- #

my_unet = build_unet(input_shape=(256,256,3), n_classes=3)
my_unet.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy',metrics=['accuracy'])
print(my_unet.summary())


history = my_unet.fit(X_train, y_train,
                    batch_size = 2,
                    verbose=1,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    shuffle=False)

#Save the model for future use
print('training done, saving the model')
my_unet.save('C:Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/premieresauvegarde.hdf5')

#Load previously saved model
from keras.models import load_model
#my_unet = load_model("C:Users/ASSMAH/Documents/Théodore/Cours/Hiver 2023/Réseaux de neurones et systemes flous/Projet/premieresauvegarde.hdf5", compile=False)
y_pred=my_unet.predict(X_test)

threshold = 0.5
test_img = X_test[0]
ground_truth=y_test[0]


test_img_input=np.expand_dims(test_img, 0)
print(test_img_input.shape)
prediction = (my_unet.predict(test_img_input)[0,:,:,0] > 0.001).astype(np.uint8)
print(prediction.shape)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth, cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()