import string
import time
import numpy as np
import os
from pickle import dump, load

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical, get_file
from tensorflow.keras.layers import Add, Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Model, load_model

from tqdm import tqdm



# load text file
def load_doc(filename):
    """Load a text file and return its contents."""
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    return text


# load image captions
def all_img_captions(filename):
    """Load captions from Flickr8k.token.txt."""

    file = load_doc(filename)

    captions = file.split("\n")
    descriptions = {}

    for caption in captions:
        if not caption.strip():
            continue

        #Each line should contain:
        #image.jpg#0    caption text
        parts = caption.split("\t", 1)

        if len(parts) != 2:
            continue

        img, caption_text = parts

        #Remove #0, #1, #2, #3, #4
        img_name = img.rsplit("#", 1)[0]

        if img_name not in descriptions:
            descriptions[img_name] = [caption_text]
        else:
            descriptions[img_name].append(caption_text)

    return descriptions



#Clean captions
def cleaning_text(captions):
    """Clean captions by removing punctuation and unwanted words."""

    table = str.maketrans("", "", string.punctuation)

    for img, caps in captions.items():

        for i, img_caption in enumerate(caps):

            #Replace hyphens with spaces
            img_caption = img_caption.replace("-", " ")

            #Split into words
            desc = img_caption.split()

            #Convert to lowercase
            desc = [word.lower() for word in desc]

            #Remove punctuation
            desc = [word.translate(table) for word in desc]

            #Remove single-character words
            desc = [word for word in desc if len(word) > 1]

            #Keep alphabetic words only
            desc = [word for word in desc if word.isalpha()]

            #Join words back together
            img_caption = " ".join(desc)

            captions[img][i] = img_caption

    return captions



#Create vocabulary

def text_vocabulary(descriptions):
    """Create a vocabulary set from all captions."""

    vocab = set()

    for key in descriptions.keys():
        for description in descriptions[key]:
            vocab.update(description.split())

    return vocab


#Save descriptions

def save_descriptions(descriptions, filename):
    """Save cleaned descriptions to a text file."""

    lines = []

    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + "\t" + desc)

    data = "\n".join(lines)

    with open(filename, "w", encoding="utf-8") as file:
        file.write(data)




#Dataset directories
dataset_text = "Flickr8k_text"
dataset_images = "Flickr8k_Dataset"

#Caption file
filename = os.path.join(dataset_text, "Flickr8k.token.txt")


#check dataset
if not os.path.exists(filename):
    print("ERROR: Caption file not found!")
    print()
    print("Expected file:")
    print(os.path.abspath(filename))
    print()
    print("Make sure Flickr8k_text/Flickr8k.token.txt exists.")
    exit(1)


#Load captions
print("Loading captions...")

descriptions = all_img_captions(filename)

print("Length of descriptions:", len(descriptions))


#clean captions
print("Cleaning captions...")

cleaned_descriptions = cleaning_text(descriptions)

print("Length of cleaned descriptions:", len(cleaned_descriptions))


#create vocabulary
print("Creating vocabulary...")

vocabulary = text_vocabulary(cleaned_descriptions)

print("Length of vocabulary:", len(vocabulary))


#save cleaned descriptions to a file
output_file = os.path.join(dataset_text, "descriptions.txt")

save_descriptions(cleaned_descriptions, output_file)

def download_with_retry(url, file_path, max_retries=3):
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            return get_file(file_path, url, cache_subdir=".", cache_dir=".")
        except Exception as e:
            if attempt < max_retries - 1:
                raise e
            print(f"Download failed on attempt failed")
            time.sleep(2 ** attempt)  # Exponential backoff

weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url, "xception_weights_tf_dim_ordering_tf_kernels_notop.h5")
model = Xception(weights=weights_path, include_top=False, pooling="avg")




def extract_features(directory):
    features = {}
    valid_images = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"] 
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0

        feature = model.predict(image)
        features[img] = feature

    return features

features = extract_features(os.path.join(dataset_images, "Flicker8k_Dataset"))
dump(features, open(os.path.join(dataset_text, "features.pkl"), "wb"))

print()
print("Cleaned descriptions saved successfully!")
print("File:", os.path.abspath(output_file))
