# Dog Breed Identification Using AI and ML

This repository contains the source code for a Flask web application that identifies the breed of a dog from an uploaded image using a pre-trained deep learning model. Additionally, the application provides information about the predicted breed, including characteristics and basic needs, retrieved from the OpenAI GPT-3 language model.

## Link to Dataset & Pre-trained Model
- [Google Drive - Dog Breed Identification](https://drive.google.com/drive/folders/1V8V6GcJHaloWTfusTd_c1GKxLswI4Clp?usp=sharing)

## Prerequisites
- Python 3.x
- Flask
- OpenAI API key

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Obtain OpenAI API Key**
    - Visit [OpenAI](https://beta.openai.com/signup/) to sign up for an account if you don't have one.
    
    - Once logged in, navigate to your account settings or dashboard to find your API key.

4. **Insert OpenAI API Key**
    - Open the `app.py` file in the project.
  
    - Locate the following line of code:
     ```python
     openai.api_key = "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"
     #Replace the placeholder API key with your actual OpenAI API key
     openai.api_key = "your-api-key-here"
     
5. **Download the Dog Identification Model:**
    - Download the pre-trained model (dog_identification.h5) from the provided [Google Drive link](https://drive.google.com/drive/folders/1V8V6GcJHaloWTfusTd_c1GKxLswI4Clp?usp=sharing).
    - Place the downloaded model file in the root directory of the project.

## Run the Application:
  ```bash
  python app.py
  The application will be accessible at http://127.0.0.1:5000/ in your web browser.
```
## Usage
Upload an Image:
    -Visit the web page and upload an image of a dog.


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
