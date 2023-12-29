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


#source path (where the Pawpularity contest data resides)
path ="C:/Users/Leonor Moura/Documents/faculdade/bioengenharia/BIOMEDICA 3/Mestrado/daco/projeto/"

#Get the metadata (the .csv data) and put it into DataFrames
train_df = pd.read_csv(path + 'train.csv')
#test_df = pd.read_csv(path + 'test.csv')

#Get the image data (the .jpg data) and put it into lists of filenames
train_jpg = glob(path + "train/*.jpg")
#test_jpg= glob(path + "test/*.jpg")       
 
luminance_features = []
saturation_features = []

for image_path in train_jpg:
    # Load image
    image = cv.imread(image_path)
    
    # Calcular luminância média
    luminance = np.mean(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
    luminance_features.append(luminance)
    
    # Calcular saturação média
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    saturation = np.mean(hsv_image[..., 1])
    saturation_features.append(saturation)
    
  

# Adicionar características de luminância e saturação ao DataFrame
train_df['luminance'] = luminance_features
train_df['saturation'] = saturation_features


# Exemplo de código para gráfico de dispersão
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