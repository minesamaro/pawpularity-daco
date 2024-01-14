# Load in packages
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
from pathlib import Path
from skimage import data, exposure
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte


def feat_extract (ids, folder="test"):
    # Get the metadata (the .csv data) and put it into DataFrames
    train_df = pd.read_csv(folder + '.csv')
    wanted_features =['Id', 'gabor_mean', 'haralick_0', 'haralick_1', 'haralick_2',
       'haralick_3', 'haralick_4', 'haralick_5', 'haralick_6', 'haralick_7',
       'haralick_8', 'haralick_9', 'haralick_10', 'haralick_11', 'haralick_12',
       'saturation', 'lbp_mean', 'total_entropy', 'mean_red', 'mean_green',
       'mean_blue', 'variance_red', 'variance_green', 'variance_blue',
       'hog_95', 'hog_49', 'hog_27', 'hog_88', 'hog_103', 'hog_28', 'hog_31',
       'hog_96', 'hog_30', 'hog_100', 'hog_90', 'hog_41', 'hog_99', 'hog_35',
       'hog_50', 'hog_36', 'hog_92', 'hog_159', 'hog_91', 'hog_19',
       'Pawpularity']
        
    gabor_features = []
    haralick_features = []
    hog_features = []
    luminance_features = []
    saturation_features = []
    lbp_features = []
    hog_feature_list = []
    total_entropy_features = []
    mean_red_features = []
    mean_green_features = []
    mean_blue_features = []
    variance_red_features = []
    variance_green_features = []
    variance_blue_features = []



    for image_path in ids:

        path = folder + "/" + image_path + ".jpg"
        # Load image
        image = cv.imread(path)
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

        # Calculate luminance
        luminance = np.mean(gray_image)
        luminance_features.append(luminance)

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

        # Extract HOG features
        fd, _ = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

        # Add HOG features to the list
        hog_feature_list.append(fd.flatten())

        ##################
        # Calculate mean values for red, green, and blue channels
        mean_red = np.mean(image[..., 2])
        mean_green = np.mean(image[..., 1])
        mean_blue = np.mean(image[..., 0])
        mean_red_features.append(mean_red)
        mean_green_features.append(mean_green)
        mean_blue_features.append(mean_blue)

        ##################
        # Calculate variance for red, green, and blue channels
        variance_red = np.var(image[..., 2])
        variance_green = np.var(image[..., 1])
        variance_blue = np.var(image[..., 0])
        variance_red_features.append(variance_red)
        variance_green_features.append(variance_green)
        variance_blue_features.append(variance_blue)

        ##################
        # Calculate entropy for red, green, and blue channels
        entropy_red = -np.sum((image[:, :, 0]/255) * np.log2((image[:, :, 0]/255) + 1e-10))
        entropy_green = -np.sum((image[:, :, 1]/255) * np.log2((image[:, :, 1]/255) + 1e-10))
        entropy_blue = -np.sum((image[:, :, 2]/255) * np.log2((image[:, :, 2]/255) + 1e-10))
        
        ##################
        # Calculate total entropy
        total_entropy = entropy_red + entropy_green + entropy_blue
        total_entropy_features.append(total_entropy)


    # Create an empty DataFrame
    final_df = pd.DataFrame()
    final_df['Id'] = ids

    # Add the "Pawpularity" column to the final DataFrame
    final_df['Pawpularity'] = train_df['Pawpularity']/100.0

    # Add Gabor features
    final_df = pd.concat([final_df, pd.DataFrame(gabor_features, columns=['gabor_mean'])], axis=1)

    # Add Haralick features
    haralick_columns = ['haralick_' + str(i) for i in range(len(haralick_features[0]))]
    final_df[haralick_columns] = haralick_features

    # Add luminance and saturation features
    final_df['luminance'] = luminance_features
    final_df['saturation'] = saturation_features

    # Add LBP features
    final_df = pd.concat([final_df, pd.DataFrame(lbp_features, columns=['lbp_mean'])], axis=1)

    # Create DataFrame with HOG features
    hog_columns = ['hog_' + str(i) for i in range(len(hog_feature_list[0]))]
    final_df[hog_columns] = pd.DataFrame(hog_feature_list)

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
        

    # select only wanted features
    final_df = final_df[wanted_features]
    # save to csv as test_features.csv
    final_df.to_csv(folder + '_features.csv', index=False)
    return final_df
            

# Get a list with the Ids of the images in the test set
test_ids = pd.read_csv('test.csv')['Id']

# Extract the features from the test set
test_image_data = feat_extract(test_ids)