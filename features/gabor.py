# Load in packages
import os
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.filters import gabor
from skimage import img_as_ubyte
from skimage import io

# Source path (where the Pawpularity contest data resides)
path = "C:/Users/Leonor Moura/Documents/faculdade/bioengenharia/BIOMEDICA 3/Mestrado/daco/projeto/"

# Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv(path + 'train.csv')

# Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob(path + "train/*.jpg")

# Gabor features list
gabor_features = []

for image_path in train_jpg:
    # Load image
    image = cv.imread(image_path)
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Resize image
    gray_image = cv.resize(gray_image, (128, 64))

    # Convert image to unsigned byte (required for Gabor filter)
    image_ubyte = img_as_ubyte(gray_image)

    # Calculate Gabor filter responses
    # You can adjust the parameters such as frequency and orientation
    gabor_responses, _ = gabor(image_ubyte, frequency=0.6, theta=1.5)

    # Extract Gabor features (mean of responses)
    gabor_mean = np.mean(gabor_responses)

    # Append Gabor features to the list
    gabor_features.append([gabor_mean])

gabor_features = np.array(gabor_features)

# Add Gabor features to the training DataFrame
gabor_columns = ['gabor_mean']
train_df[gabor_columns] = pd.DataFrame(gabor_features)

# Example code for scatter plot
plt.scatter(train_df['Pawpularity'], train_df['gabor_mean'])
plt.xlabel('Pawpularity')
plt.ylabel('Gabor Mean')
plt.title('Relationship between Gabor Mean and Pawpularity')
plt.show()
