#load in packages
import os
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2 as cv
from skimage.feature import hog
from skimage import data, exposure
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from skimage import io
import mahotas


# Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv('train.csv')

# Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob("train/*.jpg")

haralick_features = []

for image_path in train_jpg:
    # Load image
    image = cv.imread(image_path)
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Resize image
    gray_image = cv.resize(gray_image, (128, 64))
    
    # Convert image to unsigned byte (required for mahotas)
    image_ubyte = img_as_ubyte(gray_image)

    # Calculate GLCM using mahotas
    cooc_matrix = mahotas.features.haralick(image_ubyte)

    # Print the cooc_matrix to inspect its structure
    print("cooc_matrix:", cooc_matrix)

    # Extract Haralick features
    # Adjust indices based on the size of the returned array
    contrast = cooc_matrix[:, 0].mean() if cooc_matrix.shape[1] > 0 else np.nan
    correlation = cooc_matrix[:, 1].mean() if cooc_matrix.shape[1] > 1 else np.nan
    energy = cooc_matrix[:, 2].mean() if cooc_matrix.shape[1] > 2 else np.nan
    homogeneity = cooc_matrix[:, 3].mean() if cooc_matrix.shape[1] > 3 else np.nan

    # Append Haralick features to the list
    haralick_features.append([contrast, correlation, energy, homogeneity])

haralick_features = np.array(haralick_features)

# Add Haralick features to the training DataFrame
haralick_columns = ['contrast', 'correlation', 'energy', 'homogeneity']
train_df[haralick_columns] = pd.DataFrame(haralick_features)
# Example code for scatter plot
plt.scatter(train_df['Pawpularity'], train_df['contrast'])
plt.xlabel('Pawpularity')
plt.ylabel('Contrast')
plt.title('Relationship between Contrast and Pawpularity')
plt.show()

# Example code for scatter plot
plt.scatter(train_df['Pawpularity'], train_df['correlation'])
plt.xlabel('Pawpularity')
plt.ylabel('Correlation')
plt.title('Relationship between Correlation and Pawpularity')
plt.show()

# Example code for scatter plot
plt.scatter(train_df['Pawpularity'], train_df['energy'])
plt.xlabel('Pawpularity')
plt.ylabel('Energy')
plt.title('Relationship between Energy and Pawpularity')
plt.show()

# Example code for scatter plot
plt.scatter(train_df['Pawpularity'], train_df['homogeneity'])
plt.xlabel('Pawpularity')
plt.ylabel('Homogeneity')
plt.title('Relationship between Homogeneity and Pawpularity')
plt.show()