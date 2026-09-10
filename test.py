import argparse
import glob
import os
import re
from pickle import load

import numpy as np
from PIL import Image
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Arguments
# --------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Generate image caption using trained model")
    ap.add_argument(
        "-i",
        "--image",
        required=True,
        help="Path to input image"
    )
    ap.add_argument(
        "-m",
        "--model",
        default=None,
        help="Path to trained model or weights file (.keras or .h5). Default: auto-detected from models/"
    )
    ap.add_argument(
        "-t",
        "--tokenizer",
        default="Flickr8k_text/tokenizer.pkl",
        help="Path to tokenizer file (default: Flickr8k_text/tokenizer.pkl)"
    )
    return ap.parse_args()


# --------------------------------------------------
# Path resolution helper
# --------------------------------------------------

def resolve_image_path(image_path):
    """Resolve image path across common Flickr8k archive layouts."""
    if os.path.exists(image_path):
        return image_path

    filename = os.path.basename(image_path)
    candidates = [
        os.path.join("Flickr8k_Dataset", "Flicker8k_Dataset", filename),
        os.path.join("Flicker8k_Dataset", filename),
        os.path.join("Flickr8k_Dataset", filename),
        os.path.join(".", filename),
    ]

    for cand in candidates:
        if os.path.exists(cand):
            print(f"Note: '{image_path}' not found directly, resolved to '{cand}'")
            return cand

    raise FileNotFoundError(f"Image not found at '{image_path}' or any known alternative locations.")


def resolve_model_path(requested_path):
    """Find the requested or best available model in models/."""
    if requested_path:
        if os.path.exists(requested_path):
            return requested_path
        raise FileNotFoundError(f"Requested model not found: '{requested_path}'")

    # Check if metadata points to the latest trained model
    metadata_path = "models/metadata.pkl"
    if os.path.exists(metadata_path):
        try:
            metadata = load(open(metadata_path, "rb"))
            latest = metadata.get("latest_model")
            if latest and os.path.exists(latest):
                return latest
        except Exception:
            pass

    # Find models in models/ directory, prefer most recently modified
    models = glob.glob("models/model_*.*")
    if models:
        # Sort by modification time (most recently saved model first)
        models_by_mtime = sorted(models, key=os.path.getmtime, reverse=True)
        return models_by_mtime[0]

    raise FileNotFoundError("No trained model found in 'models/'. Please run main.py first to train a model.")


# --------------------------------------------------
# Extract image features
# --------------------------------------------------

def extract_features(filename, model):
    try:
        image = Image.open(filename).convert("RGB")
    except Exception as e:
        print(f"ERROR: Couldn't open image: {e}")
        return None

    image = image.resize((299, 299))
    image = np.array(image, dtype=np.float32)

    # Same preprocessing used during training
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0

    feature = model.predict(image, verbose=0)
    return feature


# --------------------------------------------------
# Find word from token ID
# --------------------------------------------------

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# --------------------------------------------------
# Generate caption
# --------------------------------------------------

def generate_desc(model, tokenizer, photo, max_length):
    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        pred = model.predict([photo, sequence], verbose=0)
        pred_id = np.argmax(pred[0])

        word = word_for_id(pred_id, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text


# --------------------------------------------------
# Model architecture definition
# --------------------------------------------------

def define_model(vocab_size, max_length):
    # Image feature branch
    inputs1 = Input(shape=(2048,), name="image_input")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    # Caption branch
    inputs2 = Input(shape=(max_length,), name="text_input")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Combine image + text
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model


def main():
    args = parse_args()

    # 1. Resolve image path
    image_path = resolve_image_path(args.image)
    print(f"Input image: {image_path}")

    # 2. Resolve model path
    model_path = resolve_model_path(args.model)
    print(f"Using model checkpoint: {model_path}")

    # 3. Load tokenizer
    tokenizer_path = args.tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run main.py first.")

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load(open(tokenizer_path, "rb"))
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary size:", vocab_size)

    # Ensure startseq/endseq are known tokens
    if "startseq" not in tokenizer.word_index or "endseq" not in tokenizer.word_index:
        print("WARNING: 'startseq' and/or 'endseq' missing from tokenizer. Please re-run main.py to re-fit the tokenizer.")

    # 4. Determine max_length
    max_length = 35  # Default with startseq + endseq tokens
    metadata_path = "models/metadata.pkl"
    if os.path.exists(metadata_path):
        try:
            metadata = load(open(metadata_path, "rb"))
            max_length = metadata.get("max_length", max_length)
        except Exception:
            pass

    # 5. Load model
    print(f"Loading caption model from {model_path}...")
    model = None
    try:
        model = load_model(model_path)
        print("Full model loaded successfully.")
        # Auto-detect max_length from model input shape if possible
        if len(model.inputs) > 1 and model.inputs[1].shape[1] is not None:
            max_length = int(model.inputs[1].shape[1])
    except Exception as e:
        print(f"Standard load_model encountered ({e}). Loading weights into architecture...")
        model = define_model(vocab_size, max_length)
        model.load_weights(model_path)
        print("Model weights loaded into architecture successfully.")

    print(f"Effective sequence max_length: {max_length}")

    # 6. Load Xception for image feature extraction
    print("Loading Xception model...")
    local_weights = "xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
    if os.path.exists(local_weights):
        print(f"Using local Xception weights: {local_weights}")
        xception_model = Xception(weights=local_weights, include_top=False, pooling="avg")
    else:
        xception_model = Xception(include_top=False, pooling="avg")
    print("Xception loaded.")

    # 7. Extract features from test image
    print("Extracting image features...")
    photo = extract_features(image_path, xception_model)
    if photo is None:
        raise SystemExit(1)

    # 8. Generate caption
    print("Generating caption...")
    description = generate_desc(model, tokenizer, photo, max_length)

    # Clean start and end tokens for clean display
    cleaned_caption = description.replace("startseq", "").replace("endseq", "").strip()
    # Normalize multiple spaces
    cleaned_caption = " ".join(cleaned_caption.split())

    print("\n" + "=" * 50)
    print("Generated Caption:")
    print(cleaned_caption if cleaned_caption else "[No caption generated]")
    print("=" * 50)


if __name__ == "__main__":
    main()
