import argparse
import os
import string
import time
from pickle import dump, load

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import get_file, to_categorical
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

        # Each line contains: image.jpg#0    caption text
        parts = caption.split("\t", 1)
        if len(parts) != 2:
            continue

        img, caption_text = parts
        # Remove #0, #1, #2, #3, #4
        img_name = img.rsplit("#", 1)[0]

        if img_name not in descriptions:
            descriptions[img_name] = [caption_text]
        else:
            descriptions[img_name].append(caption_text)

    return descriptions


# Clean captions
def cleaning_text(captions):
    """Clean captions by removing punctuation and unwanted words."""
    table = str.maketrans("", "", string.punctuation)

    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            # Replace hyphens with spaces
            img_caption = img_caption.replace("-", " ")

            # Split into words
            desc = img_caption.split()

            # Convert to lowercase
            desc = [word.lower() for word in desc]

            # Remove punctuation
            desc = [word.translate(table) for word in desc]

            # Remove single-character words
            desc = [word for word in desc if len(word) > 1]

            # Keep alphabetic words only
            desc = [word for word in desc if word.isalpha()]

            # Join words back together
            img_caption = " ".join(desc)
            captions[img][i] = img_caption

    return captions


# Create vocabulary
def text_vocabulary(descriptions):
    """Create a vocabulary set from all captions."""
    vocab = set()
    for key in descriptions.keys():
        for description in descriptions[key]:
            vocab.update(description.split())
    return vocab


# Save descriptions
def save_descriptions(descriptions, filename):
    """Save cleaned descriptions to a text file."""
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + "\t" + desc)

    data = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(data)


def download_with_retry(url, file_path, max_retries=3):
    """Download a file with retry logic or use local file if present."""
    if os.path.exists(file_path):
        print(f"Using existing local weights: {file_path}")
        return file_path
    for attempt in range(max_retries):
        try:
            return get_file(file_path, url, cache_subdir=".", cache_dir=".")
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Download failed on attempt {attempt + 1}, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff


def extract_features(directory, xception_model):
    """Extract 2048-d feature vectors from images in directory using Xception."""
    features = {}
    valid_images = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for img in tqdm(os.listdir(directory), desc="Extracting features"):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = os.path.join(directory, img)
        try:
            image = Image.open(filename).convert("RGB")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        image = image.resize((299, 299))
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0

        feature = xception_model.predict(image, verbose=0)
        features[img] = feature

    return features


def load_photos(filename, available_features):
    """Load image filenames from a split file that exist in features."""
    file = load_doc(filename)
    photos = [p.strip() for p in file.split("\n") if p.strip()]
    photos_present = [photo for photo in photos if photo in available_features]
    return photos_present


def load_clean_descriptions(filename, photos):
    """Load cleaned descriptions and wrap with startseq and endseq tokens."""
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        tokens = line.split("\t")
        if len(tokens) < 2:
            continue
        image_id, image_caption = tokens[0], tokens[1].strip()
        if image_id in photos:
            if image_id not in descriptions:
                descriptions[image_id] = []
            # Wrap with startseq and endseq for training sequence generation
            desc = f"startseq {image_caption} endseq"
            descriptions[image_id].append(desc)
    return descriptions


def load_features(filename, photos):
    """Load cached features for a set of photos."""
    all_features = load(open(filename, "rb"))
    features = {k: all_features[k] for k in photos if k in all_features}
    print("Length of features loaded for split:", len(features))
    return features


def dict_to_list(descriptions):
    """Convert a dictionary of descriptions to a list of strings."""
    all_desc = []
    for key in descriptions.keys():
        for d in descriptions[key]:
            all_desc.append(d)
    return all_desc


def create_tokenizer(descriptions):
    """Create a tokenizer from the descriptions."""
    lines = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def calc_max_length(descriptions):
    """Calculate the maximum length of descriptions in words."""
    lines = dict_to_list(descriptions)
    return max(len(d.split()) for d in lines)


def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    """Generate batches of ([image_feature, in_seq], out_seq) for model training."""
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            if key not in features:
                continue
            feature = features[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        X2_padded = pad_sequences(X2, maxlen=max_length)
                        y_cat = to_categorical(y, num_classes=vocab_size)
                        yield (np.array(X1, dtype=np.float32), np.array(X2_padded, dtype=np.int32)), np.array(y_cat, dtype=np.float32)
                        X1, X2, y = [], [], []
                        n = 0


def define_model(vocab_size, max_length):
    """Define the image captioning model architecture."""
    # CNN image feature branch: 2048 -> 256
    inputs1 = Input(shape=(2048,), name="image_input")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    # LSTM caption sequence branch: max_length -> 256
    inputs2 = Input(shape=(max_length,), name="text_input")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model: Combine image + text representations
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    # Tie inputs and outputs together
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Flickr8k Image Captioning Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--steps-per-epoch", type=int, default=5, help="Steps per epoch (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--extract-features", action="store_true", help="Force re-extraction of image features with Xception")
    args = parser.parse_args()

    # Dataset directories
    dataset_text = "Flickr8k_text"
    dataset_images = "Flickr8k_Dataset"

    # Caption file
    token_filename = os.path.join(dataset_text, "Flickr8k.token.txt")

    # Check dataset
    if not os.path.exists(token_filename):
        print("ERROR: Caption file not found!")
        print("Expected file:", os.path.abspath(token_filename))
        print("Make sure Flickr8k_text/Flickr8k.token.txt exists.")
        exit(1)

    # Step 1: Load and clean captions
    print("Loading raw captions from:", token_filename)
    descriptions = all_img_captions(token_filename)
    print("Length of raw descriptions:", len(descriptions))

    print("Cleaning captions...")
    cleaned_descriptions = cleaning_text(descriptions)
    print("Length of cleaned descriptions:", len(cleaned_descriptions))

    vocabulary = text_vocabulary(cleaned_descriptions)
    print("Vocabulary size (raw words):", len(vocabulary))

    descriptions_output_file = os.path.join(dataset_text, "descriptions.txt")
    save_descriptions(cleaned_descriptions, descriptions_output_file)
    print("Cleaned descriptions saved to:", os.path.abspath(descriptions_output_file))

    # Step 2: Extract or Load Features
    features_path = os.path.join(dataset_text, "features.pkl")
    if os.path.exists(features_path) and not args.extract_features:
        print(f"\nExisting features found at {features_path}. Loading cached features...")
        features = load(open(features_path, "rb"))
        print(f"Loaded features for {len(features)} images.")
    else:
        print("\nExtracting features using Xception...")
        weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
        local_weights = "xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
        weights_path = download_with_retry(weights_url, local_weights)
        xception_model = Xception(weights=weights_path, include_top=False, pooling="avg")

        # Check directory layout: Flickr8k_Dataset/Flicker8k_Dataset or Flickr8k_Dataset
        img_dir = os.path.join(dataset_images, "Flicker8k_Dataset")
        if not os.path.exists(img_dir):
            img_dir = dataset_images

        features = extract_features(img_dir, xception_model)
        dump(features, open(features_path, "wb"))
        print(f"Extracted and saved features for {len(features)} images to {features_path}.")

    # Step 3: Load train split photos, captions, and features
    train_split_file = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
    train_images = load_photos(train_split_file, features)
    print(f"\nPhotos in train split matching features: {len(train_images)}")

    train_descriptions = load_clean_descriptions(descriptions_output_file, train_images)
    print(f"Descriptions loaded for train photos: {len(train_descriptions)}")

    train_features = load_features(features_path, train_images)

    # Step 4: Tokenizer with startseq and endseq
    print("\nCreating tokenizer on training descriptions...")
    train_tokenizer = create_tokenizer(train_descriptions)
    tokenizer_path = os.path.join(dataset_text, "tokenizer.pkl")
    dump(train_tokenizer, open(tokenizer_path, "wb"))
    print(f"Tokenizer saved to {tokenizer_path}.")

    vocab_size = len(train_tokenizer.word_index) + 1
    print("Vocabulary size (including startseq and endseq):", vocab_size)
    print("Special token 'startseq' ID:", train_tokenizer.word_index.get("startseq"))
    print("Special token 'endseq' ID:", train_tokenizer.word_index.get("endseq"))

    max_length = calc_max_length(train_descriptions)
    print("Maximum length of descriptions:", max_length)

    # Step 5: Test data generator batch
    print("\nTesting batch data generator...")
    test_gen = data_generator(train_descriptions, train_features, train_tokenizer, max_length, vocab_size, batch_size=args.batch_size)
    (sample_x, sample_y) = next(test_gen)
    print("Batch Input 1 shape (image features):", sample_x[0].shape)
    print("Batch Input 2 shape (text sequences):", sample_x[1].shape)
    print("Batch Output shape (targets):", sample_y.shape)

    # Step 6: Define and compile model
    print("\nDefining caption model...")
    model = define_model(vocab_size, max_length)
    model.summary()

    # Step 7: Train and save model checkpoints
    os.makedirs("models", exist_ok=True)
    train_gen = data_generator(train_descriptions, train_features, train_tokenizer, max_length, vocab_size, batch_size=args.batch_size)

    print(f"\nStarting training: {args.epochs} epochs, {args.steps_per_epoch} steps per epoch...")
    latest_keras_path = None
    for i in range(args.epochs):
        print(f"\n--- Epoch {i + 1}/{args.epochs} ---")
        model.fit(train_gen, epochs=1, steps_per_epoch=args.steps_per_epoch, verbose=1)
        keras_path = f"models/model_{i + 1}.keras"
        h5_path = f"models/model_{i + 1}.h5"
        model.save(keras_path)
        model.save(h5_path)
        latest_keras_path = keras_path
        print(f"Saved checkpoint: {keras_path} and {h5_path}")

    # Save training metadata for test.py
    metadata = {
        "max_length": max_length,
        "vocab_size": vocab_size,
        "latest_model": latest_keras_path,
    }
    dump(metadata, open(os.path.join("models", "metadata.pkl"), "wb"))

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
