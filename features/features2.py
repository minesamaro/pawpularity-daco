# Load in packages
import os
import seaborn as sns
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Source path (where the Pawpularity contest data resides)
path = "C:/Users/Leonor Moura/Documents/faculdade/bioengenharia/BIOMEDICA 3/Mestrado/daco/projeto/"

# Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv(path + 'train.csv')

# Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob(path + "train/*.jpg")

# Features lists

total_entropy_features = []
mean_red_features = []
mean_green_features = []
mean_blue_features = []
variance_red_features = []
variance_green_features = []
variance_blue_features = []

# Assuming selected_features is a list of column names you want to visualize from your DataFrame
selected_features = ['Id','total_entropy', 'mean_red', 'mean_green', 'mean_blue', 'variance_red', 'variance_green', 'variance_blue']

for image_path in train_jpg:
    # Load image
    image = cv.imread(image_path)

    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Resize image
    gray_image = cv.resize(gray_image, (128, 64))

   

    # Calculate mean values for red, green, and blue channels
    mean_red = np.mean(image[..., 2])
    mean_green = np.mean(image[..., 1])
    mean_blue = np.mean(image[..., 0])
    mean_red_features.append(mean_red)
    mean_green_features.append(mean_green)
    mean_blue_features.append(mean_blue)

    # Calculate variance for red, green, and blue channels
    variance_red = np.var(image[..., 2])
    variance_green = np.var(image[..., 1])
    variance_blue = np.var(image[..., 0])
    variance_red_features.append(variance_red)
    variance_green_features.append(variance_green)
    variance_blue_features.append(variance_blue)

    # Calculate entropy for red, green, and blue channels
    entropy_red = -np.sum((image[:, :, 0]/255) * np.log2((image[:, :, 0]/255) + 1e-10))
    entropy_green = -np.sum((image[:, :, 1]/255) * np.log2((image[:, :, 1]/255) + 1e-10))
    entropy_blue = -np.sum((image[:, :, 2]/255) * np.log2((image[:, :, 2]/255) + 1e-10))
    
    # Calculate total entropy
    total_entropy = entropy_red + entropy_green + entropy_blue
    total_entropy_features.append(total_entropy)

# Create an empty DataFrame
final_df = pd.DataFrame()

# Add the "Pawpularity" column to the final DataFrame
final_df['Id'] = train_df['Id']
final_df['Pawpularity'] = train_df['Pawpularity']



# Add total entropy feature
final_df = pd.concat([final_df, pd.DataFrame(total_entropy_features, columns=['total_entropy'])], axis=1)

# Add mean values for red, green, and blue channels
final_df['mean_red'] = mean_red_features
final_df['mean_green'] = mean_green_features
final_df['mean_blue'] = mean_blue_features

# Add variance for red, green, and blue channels
final_df['variance_red'] = variance_red_features
final_df['variance_green'] = variance_green_features
final_df['variance_blue'] = variance_blue_features

# Calculate the correlation matrix
#correlation_matrix = final_df.corr()

# Check if selected features have non-null values
#print("Null values in selected features:")
#print(final_df[selected_features].isnull().sum())

# Add the correlation matrix to the final DataFrame
#final_df = pd.concat([final_df, correlation_matrix['Pawpularity']], axis=1, keys=['features', 'correlation'])

# Display the final DataFrame
#print(final_df.head())

# Plot the correlation matrix
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
#plt.title('Correlation Matrix')
#plt.show()