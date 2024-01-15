
import os
import seaborn as sns
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.filters import gabor
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern
import mahotas
from skimage.feature import hog


# Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv('train.csv')

# Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob("train/*.jpg")

# Features lists
gabor_features = []
haralick_features = []
hog_features = []

saturation_features = []
lbp_features = []
# Selected_features 
selected_features = ['Id','gabor_mean', 'haralick_0', 'haralick_1', 'haralick_2', 'haralick_3', 'saturation', 'lbp_mean']

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
    gabor_responses, _ = gabor(image_ubyte, frequency=0.6, theta=1.5)
    # Extract Gabor features (mean of responses)
    gabor_mean = np.mean(gabor_responses)
    gabor_features.append(gabor_mean)

    # Calculate GLCM using mahotas
    cooc_matrix = mahotas.features.haralick(image_ubyte)

    # Extract Haralick features
    # Adjust indices based on the size of the returned array
    if cooc_matrix.shape[1] >= 4:
        haralick_features.append(np.mean(cooc_matrix, axis=0))
    else:
        haralick_features.append([np.nan] * 4)

    # Extract HOG features
    fd, _ = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    hog_features.extend(fd)

  

    # Convert image to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Calculate saturation
    saturation = np.mean(hsv_image[..., 1])
    saturation_features.append(saturation)


    # Calculate LBP features
    lbp = local_binary_pattern(image_ubyte, P=8, R=5, method="uniform")
    # Extract LBP features (mean of LBP)
    lbp_mean = np.mean(lbp)
    lbp_features.append(lbp_mean)

# Create an empty DataFrame
final_df = pd.DataFrame()

# Add Id and Pawpularity columns
final_df['Id'] = train_df['Id']
final_df['Pawpularity'] = train_df['Pawpularity']

# Add Gabor features
final_df = pd.concat([final_df, pd.DataFrame(gabor_features, columns=['gabor_mean'])], axis=1)

# Add Haralick features
haralick_columns = ['haralick_' + str(i) for i in range(len(haralick_features[0]))]
final_df[haralick_columns] = haralick_features

# Add saturation features
final_df['saturation'] = saturation_features

# Add LBP features
final_df = pd.concat([final_df, pd.DataFrame(lbp_features, columns=['lbp_mean'])], axis=1)

# Calculate the correlation matrix
correlation_matrix = final_df.corr()

# Check if selected features have non-null values
print("Null values in selected features:")
print(final_df[selected_features].isnull().sum())

# Add the correlation matrix to the final DataFrame
final_df = pd.concat([final_df, correlation_matrix['Pawpularity']], axis=1, keys=['features', 'correlation'])

# Display the final DataFrame
print(final_df.head())

# Plot the correlation matrix
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()