#load in packages
import os
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



        
#source path (where the Pawpularity contest data resides)
path ="C:/Users/Leonor Moura/Documents/faculdade/bioengenharia/BIOMEDICA 3/Mestrado/daco/projeto/"

#Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv(path + 'train.csv')
#test_df = pd.read_csv(path + 'test.csv')

#Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob(path + "train/*.jpg")
#test_jpg= glob(path + "test/*.jpg")       
 
#extract features from image
import cv2 as cv
from skimage.feature import hog
from skimage import data, exposure

# Example code for extracting HOG features for all images in the training set

hog_features = []

for image_path in train_jpg[0:50]:
    # Load image
    image = cv.imread(image_path)
    # Convert to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Resize image
    image = cv.resize(image, (128, 64))
    # Extract HOG features
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)
    # Append HOG features to the list
    hog_features.append(fd)

hog_features = np.array(hog_features)
# Add HOG features to the training DataFrame using pd.concat(axis=1)
hog_columns = [f'hog_{i}' for i in range(hog_features.shape[1])]
train_df = pd.concat([train_df, pd.DataFrame(hog_features, columns=hog_columns)], axis=1)

# Example code for scatter plot
for hog_column in hog_columns:
    plt.scatter(train_df['Pawpularity'], train_df[hog_column], label=hog_column)


plt.xlabel('Pawpularity')
plt.ylabel('HOG Feature')
plt.title('Relationship between HOG Feature and Pawpularity')
plt.show()