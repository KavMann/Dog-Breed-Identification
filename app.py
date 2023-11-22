from flask import Flask, render_template, request, redirect, url_for
import http.client
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import pandas as pd
import os
import openai

openai.api_key = "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"

app = Flask(__name__)

# Load the pre-trained model
model = load_model("dog_identification.h5")

csv_file_path = os.path.join(os.path.dirname(__file__), "labels.csv")
colnames = ['Id', 'breed']
# Read the CSV file
df_labels = pd.read_csv(csv_file_path, names=colnames)

# Define new_list in a global scope
num_breeds = 60
breed_dict = list(df_labels['breed'].value_counts().keys())
new_list = sorted(breed_dict, reverse=True)[:num_breeds * 2 + 1:2]

# Check if the prediction confidence is below a threshold
def is_dog(pred_val, confidence_threshold=0.5):
    if np.max(pred_val) < confidence_threshold:
        return False
    return True

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_breed = None
    breed_info = None
    image_path = None

    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        # Save the uploaded file temporarily
        image_path = "uploaded_image.jpg"
        file.save(image_path)

        # Predict the breed
        pred_val = predict_breed(image_path)
        predicted_breed = sorted(new_list)[np.argmax(pred_val)]

        # Get breed information from ChatGPT
        breed_info = get_breed_characteristics(predicted_breed)
    return render_template("index.html", predicted_breed=predicted_breed, breed_info=breed_info, image_path=image_path)

def predict_breed(image_path):
    img_array = cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), (224, 224))
    img_array = preprocess_input(np.expand_dims(np.array(img_array[...,::-1].astype(np.float32)).copy(), axis=0))
    pred_val = model.predict(np.array(img_array, dtype="float32"))
    return pred_val

def get_breed_characteristics(breed):
    # Define the prompt
    prompt = f"What are the basic characteristics of the {breed} dog breed?"

    # Call the OpenAI API to get a response
    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose an appropriate engine
        prompt=prompt,
        max_tokens=250,  # Adjust the max_tokens as needed
        temperature=0.7,  # Adjust the temperature as needed
        stop=None  # You can specify stop words to end the response
    )

    return response.choices[0].text

def get_breed_needs(breed):
    # Define the prompt for basic needs
    prompt2 = f"What are the basic needs of the {breed} dog breed?"

    # Call the OpenAI API to get a response
    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose an appropriate engine
        prompt=prompt2,
        max_tokens=250,  # Adjust the max_tokens as needed
        temperature=0.7,  # Adjust the temperature as needed
        stop=None  # You can specify stop words to end the response
    )

    return response.choices[0].text

@app.route('/<path:path>')
def static_css(path):
    return app.send_static_file(path)

if __name__ == "__main__":
    app.run(debug=True)
