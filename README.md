# Dog-Breed-Identification Using AI and ML
This repository contains the source code for a Flask web application that identifies the breed of a dog from an uploaded image using a pre-trained deep learning model. Additionally, the application provides information about the predicted breed, including characteristics and basic needs, retrieved from the OpenAI GPT-3 language model.

## Features
- **Dog Breed Identification:** The application uses a pre-trained deep learning model to predict the breed of a dog based on an uploaded image.

- **Breed Information:** The predicted breed's characteristics are fetched from the OpenAI GPT-3 language model to provide users with basic information about the breed.

- **Breed Needs:** The application also queries the GPT-3 model for information about the basic needs of the predicted dog breed, assisting users in understanding the requirements of the breed before adoption.

## Dataset
The training data is sourced from a CSV file (labels.csv) containing information about dog breeds and corresponding image file names. The dataset is preprocessed to include only a specified number of unique dog breeds.

## Model Architecture
The model is built using the InceptionV3 architecture, a pre-trained model on the ImageNet dataset. The top layers of InceptionV3 are modified to suit the specific task of dog breed identification. Batch normalization, global average pooling, and dropout layers are added to enhance the model's performance.

## Training
The model is trained using an ImageDataGenerator for data augmentation. The training and testing datasets are split in an 80:20 ratio. The compiled model is trained using the RMSprop optimizer and sparse categorical crossentropy loss function.

## Prediction
The trained model is saved as "dog_identification.h5" and can be loaded for making predictions. The provided code demonstrates how to load the model, preprocess an image, and predict the dog breed. The prediction includes checking if the confidence of the prediction is above a specified threshold.
