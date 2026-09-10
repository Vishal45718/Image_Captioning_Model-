# Neural Image Captioning with Xception and LSTM

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B%20%2F%202.21-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Flickr8k-green.svg)](https://forms.illinois.edu/sec/1713398)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end deep learning system that automatically generates descriptive natural language captions for input photographs. The project combines **Convolutional Neural Networks (CNN)** for computer vision feature representation and **Recurrent Neural Networks (LSTM)** for autoregressive sequence modeling, trained on the classic **Flickr8k** benchmark.

---

## Architecture Overview

This project implements a multi-modal **Encoder-Decoder Merge Architecture** inspired by standard neural image captioning research:

```
               +---------------------------+
               |        Input Image        |
               +-------------+-------------+
                             |
                             v
               +---------------------------+
               |  Xception (Pretrained)    |
               |  Global Avg Pooling Layer |
               +-------------+-------------+
                             |
                   Image Vector (2048-d)
                             |
                             v
               +---------------------------+
               |  Dropout(0.5) + Dense     |
               |       (256 units)         |
               +-------------+-------------+
                             |
                             |
                             +-------------------+
                                                 |
                                                 v
+------------------+                   +--------------------+
| Partial Caption  |                   |  Element-wise Add  |
|  (Token IDs)     |                   +---------+----------+
+--------+---------+                             |
         |                                       v
         v                             +--------------------+
+------------------+                   |    Dense (256)     |
| Embedding (256)  |                   |     ReLU Act.      |
|  mask_zero=True  |                   +---------+----------+
+--------+---------+                             |
         |                                       v
         v                             +--------------------+
+------------------+                   | Dense(vocab_size)  |
|   Dropout(0.5)   |                   |    Softmax Act.    |
+--------+---------+                   +---------+----------+
         |                                       |
         v                                       v
+------------------+                   +--------------------+
|    LSTM(256)     |------------------>| Next Word Predict. |
+------------------+                   +--------------------+
```

1. **Vision Encoder (Xception)**: Extracts rich semantic feature embeddings ($1 \times 2048$) from raw $299 \times 299$ images.
2. **Text Decoder (Embedding + LSTM)**: Ingests the history of emitted words and maintains contextual sequence state.
3. **Merge Head**: Combines image and language representations in a common 256-dimensional space and predicts a probability distribution over the vocabulary.

---

## Repository Structure

```text
Image_Captioning_Model/
├── Flickr8k_Dataset/
│   ├── 1859941832_7faf6e5fa9.jpg -> Flicker8k_Dataset/...  (Symlinked convenience path)
│   └── Flicker8k_Dataset/         # 8,091 Flickr images (.jpg)
├── Flickr8k_text/
│   ├── Flickr8k.token.txt         # 40,455 raw caption annotations (5 per image)
│   ├── Flickr_8k.trainImages.txt  # Training split list (6,000 images)
│   ├── Flickr_8k.devImages.txt    # Validation split list (1,000 images)
│   ├── Flickr_8k.testImages.txt   # Test split list (1,000 images)
│   ├── descriptions.txt           # Cleaned & tokenized text corpus
│   ├── features.pkl               # Cached 2048-d Xception features (~66 MB)
│   └── tokenizer.pkl              # Fitted Tokenizer with startseq / endseq tokens
├── models/
│   ├── metadata.pkl               # Runtime config (max_length, vocab_size, latest model)
│   ├── model_*.keras              # Modern native Keras 3 format checkpoints
│   └── model_*.h5                 # Backward-compatible HDF5 checkpoints
├── main.py                        # Full training & data preparation pipeline
├── test.py                        # Single-image caption generation (inference)
├── README.md                      # Project documentation
└── xception_weights_tf_dim_ordering_tf_kernels_notop.h5 # Pretrained Xception weights
```

> **Note on Dataset Folder Naming**:  
> In the official Flickr8k archive release, the internal folder is spelled `Flicker8k_Dataset` (with an `"er"`). The pipeline automatically handles this discrepancy with fallback path resolution, so you can pass either `Flickr8k_Dataset/<image>.jpg` or `Flickr8k_Dataset/Flicker8k_Dataset/<image>.jpg`.

---

## Getting Started

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Vishal45718/Image_Captioning_Model-.git
cd Image_Captioning_Model-

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install tensorflow keras pillow tqdm numpy matplotlib
```

### 2. Dataset Preparation

Download the **Flickr8k Dataset** and **Flickr8k Text** archives (e.g. from Kaggle or the official release) and place them in the project root:
- Extract images to `Flickr8k_Dataset/`
- Extract text annotations to `Flickr8k_text/`

Ensure the pretrained Xception weights file (`xception_weights_tf_dim_ordering_tf_kernels_notop.h5`) is placed in the project root or let `main.py` download it automatically.

---

## Training the Model

To run the complete data preparation and training pipeline:

```bash
python main.py
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `10` | Total number of training epochs |
| `--steps-per-epoch` | `5` | Gradient steps per epoch (increase for production training) |
| `--batch-size` | `32` | Batch size for sequence generation |
| `--extract-features` | `False` | Force re-extraction of image features even if `features.pkl` exists |

### Training Modes

- **Smoke Test / Fast Validation**:
  ```bash
  python main.py --epochs 5 --steps-per-epoch 50
  ```
  Runs in under a minute on CPU to verify that feature loading, tokenization, sequence batching, and model checkpointing work properly.

- **Full Training (GPU Recommended)**:
  ```bash
  python main.py --epochs 25 --steps-per-epoch 1000 --batch-size 64
  ```
  Trains thoroughly over the 30,000 training captions to minimize categorical cross-entropy loss down to fluent caption generation.

---

## Inference (Caption Generation)

Once a model checkpoint is saved, test it on any image using `test.py`:

```bash
python test.py --image Flickr8k_Dataset/1859941832_7faf6e5fa9.jpg
```

### Output Example:
```text
Input image: Flickr8k_Dataset/1859941832_7faf6e5fa9.jpg
Using model checkpoint: models/model_5.keras
Loading tokenizer from Flickr8k_text/tokenizer.pkl...
Vocabulary size: 7320
Loading caption model from models/model_5.keras...
Full model loaded successfully.
Effective sequence max_length: 35
Loading Xception model...
Using local Xception weights: xception_weights_tf_dim_ordering_tf_kernels_notop.h5
Xception loaded.
Extracting image features...
Generating caption...

==================================================
Generated Caption:
dog in in the
==================================================
```

### Custom Model or Image
You can pass any trained checkpoint explicitly using `--model`:

```bash
python test.py --image Flickr8k_Dataset/1000268201_693b08cb0e.jpg --model models/model_5.keras
```

Or test with legacy `.h5` files:
```bash
python test.py --image Flickr8k_Dataset/1859941832_7faf6e5fa9.jpg --model models/model_5.h5
```

---

## Engineering Details & Fixes Implemented

This codebase has been thoroughly audited and modernized for **TensorFlow 2.16+ and Keras 3**:

- **Sequence Boundary Framing**: Training captions are wrapped with `startseq` and `endseq`. Without these tokens, autoregressive generation cannot initiate or terminate properly.
- **Keras 3 Deserialization**: Resolves the `Unknown layer: 'NotEqual'` issue when loading models using `mask_zero=True` in `Embedding` layers by providing both native `.keras` serialization and architecture-based weight loading for legacy `.h5` files.
- **Feature Extraction Caching**: Detects pre-extracted bottleneck representations in `features.pkl` to prevent redundant multi-hour re-extraction runs on CPU.
- **Vectorized Sequence Generator**: Yields batches of padded token sequences and one-hot categorical matrices for high-throughput training.
- **Flexible Path Resolution**: Intelligently resolves image files regardless of whether dataset folders are named `Flickr8k_Dataset` or `Flicker8k_Dataset`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Hodosh, Young, and Hockenmaier (2013)** for the [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398).
- **François Chollet** for the [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) architecture.
- **Show and Tell: A Neural Image Caption Generator** (Vinyals et al., 2015) for inspiring the encoder-decoder captioning paradigm.
