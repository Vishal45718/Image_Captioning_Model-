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
            print(f"Download failed on attempt {attempt + 1}")
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


def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if photo in features]
    return photos_present


def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        tokens = line.split("\t")
        if len(tokens) < 2:
            continue
        image_id, image_caption = tokens[0], tokens[1]
        if image_id in photos:
            if image_id not in descriptions:
                descriptions[image_id] = []
            descriptions[image_id].append(image_caption)
    return descriptions

def load_features(filename, photos):
    all_features = load(open(filename, "rb"))
    features = {k: all_features[k] for k in photos if k in all_features}
    print("Length of features:", len(features))
    return features

filename = dataset_text + "/Flickr_8k.trainImages.txt"
train_images = load_photos(filename)
train_descriptions = load_clean_descriptions(os.path.join(dataset_text, "descriptions.txt"), train_images)
train_features = load_features(os.path.join(dataset_text, "features.pkl"), train_images)



def dict_to_list(descriptions):
    """Convert a dictionary of descriptions to a list of strings."""
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    """Create a tokenizer from the descriptions."""
    lines = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer



train_descriptions_list = dict_to_list(train_descriptions)
train_tokenizer = create_tokenizer(train_descriptions)

dump(train_tokenizer, open(os.path.join(dataset_text, "tokenizer.pkl"), "wb")) 

vocab_size = len(train_tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)

def max_length(descriptions):
    """Calculate the maximum length of descriptions."""
    lines = dict_to_list(descriptions)
    return max(len(d.split()) for d in lines)

max_length = max_length(train_descriptions)
print("Maximum length of descriptions:", max_length)

def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    """Generate batches of data for training."""
    while True:
        for key, desc_list in descriptions.items():
            feature = features[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [feature, in_seq], out_seq
    output_signature = (
        {
            "input_1": tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            "input_2": tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    ) 


    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(descriptions, features, tokenizer, max_length, vocab_size),
        output_signature=output_signature
    )

    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


def create_sequence(tokenizer, max_length, desc_list, feature):
    """Create input-output sequences for a given image feature and its descriptions."""
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


dataset = data_generator(train_descriptions, train_features, train_tokenizer, max_length, vocab_size)

for (a, b) in dataset.take(1):
    print("Input 1 shape:", a["input_1"].shape)
    print("Input 2 shape:", a["input_2"].shape)
    print("Output shape:", b.shape)
    break

def define_model(vocab_size, max_length):
    """Define the image captioning model architecture."""
    #CNN model from 2048 nodes to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    #LSTM model from max_length to 256 nodes
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    # Tie it together [image + caption] -> [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    return model

print()
print("Cleaned descriptions saved successfully!")
print("File:", os.path.abspath(output_file))
