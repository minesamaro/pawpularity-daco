# Load in packages
import os
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte


# Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv('train.csv')

# Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob("train/*.jpg")

# LBP features list
lbp_features = []

for image_path in train_jpg:
    # Load image
    image = cv.imread(image_path)
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Resize image
    gray_image = cv.resize(gray_image, (128, 64))

    # Convert image to unsigned byte (required for LBP)
    image_ubyte = img_as_ubyte(gray_image)

    # Calculate LBP features
    # You can adjust the parameters such as P (number of neighbors) and R (radius)
    lbp = local_binary_pattern(image_ubyte, P=8, R=5, method="uniform")

    # Extract LBP features (mean of LBP)
    lbp_mean = np.mean(lbp)

    # Append LBP features to the list
    lbp_features.append([lbp_mean])

lbp_features = np.array(lbp_features)

# Add LBP features to the training DataFrame
lbp_columns = ['lbp_mean']
train_df[lbp_columns] = pd.DataFrame(lbp_features)

# Example code for scatter plot
plt.scatter(train_df['Pawpularity'], train_df['lbp_mean'])
plt.xlabel('Pawpularity')
plt.ylabel('LBP Mean')
plt.title('Relationship between LBP Mean and Pawpularity')
plt.show()
