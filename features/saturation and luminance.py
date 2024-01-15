import cv2 as cv
import torch
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from skimage.feature import hog
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv('train.csv')

# Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob("train/*.jpg")
  
 
luminance_features = []
saturation_features = []

for image_path in train_jpg:
    # Load image
    image = cv.imread(image_path)
    
    # calculate luminance
    luminance = np.mean(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
    luminance_features.append(luminance)
    
    # calculate saturation
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    saturation = np.mean(hsv_image[..., 1])
    saturation_features.append(saturation)
    
  

# Add luminance and saturation features to the training DataFrame
train_df['luminance'] = luminance_features
train_df['saturation'] = saturation_features


# Example code for scatter plot
plt.scatter(train_df['Pawpularity'], train_df['luminance'])
plt.xlabel('Pawpularity')
plt.ylabel('Luminance')
plt.title('Relationship between Luminance and Pawpularity')
plt.show()

plt.scatter(train_df['Pawpularity'], train_df['saturation'])
plt.xlabel('Pawpularity')
plt.ylabel('Saturation')
plt.title('Relationship between Saturation and Pawpularity')
plt.show()