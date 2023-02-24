# Text Generation with PyTorch on Large Movie Review Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IoM2-M1Hjv1N4OVTzZHvIh_WfMe1HRjZ?usp=sharing)

This repository contains a deep learning model for text generation using PyTorch on the Large Movie Review Dataset. The model is trained to generate movie reviews in a language that mimics the style and sentiment of the reviews in the original dataset.

The example notebook demonstrates how to train the model and generate text using the decoder-only architecture. The model can be trained on a GPU using the `train.py` script. The `run.py` script can be used to generate text using a pre-trained model.

## Table of Contents

-   [Installation](#installation)
-   [Dataset](#dataset)
-   [Folder Structure](#folder-structure)
-   [Usage](#usage)
-   [Model](#model)
-   [Model Evaluation](#model-evaluation)
-   [Demo](#demo)
-   [References](#references)
-   [Contributing](#contributing)
-   [License](#license)

## Installation

To use this repository, you will need to have PyTorch installed
with Python 3.6 or higher. All the dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The Large Movie Review Dataset consists of 50,000 movie reviews from IMDB, labeled as positive or negative. The dataset is preprocessed and split into a training set of 25,000 reviews and a test set of 25,000 reviews. The dataset can be downloaded from the [IMDB website](http://ai.stanford.edu/~amaas/data/sentiment/).

## Folder Structure

The repository contains the following files and folders:

```text
    ./
    ├── aclImdb/
    |   ├── test/
    |   |   ├── neg/
    |   |   └── pos/
    |   └── train/
    |       ├── neg/
    |       └── pos/
    ├── models/
    |   ├── model.pt
    |   └── tokenizer.json
    ├── .gitattributes
    ├── .gitignore
    ├── dataset.py
    ├── model.py
    ├── notebook.ipynb
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── run.py
    ├── tokenizer.py
    ├── train.py
    └── utils.py
```

## Model

The text generation model is based on a transformer decoder architecture with multi-head attention and positional encoding. The model is implemented using PyTorch and trained on the preprocessed dataset using Adam optimizer with backpropagation. The model is designed to generate text one word at a time, conditioned on a given starting word or sequence of words.

### Model Architecture

The model consists of a learned embedding layer to map the input words to a continuous vector space, followed by multiple transformer decoder layers to capture long-term dependencies and generate output words. The decoder layers include multi-head attention, which allows the model to attend to different parts of the input sequence, and positional encoding, which encodes the position of each word in the input sequence. The model is trained to minimize the cross-entropy loss between the predicted and actual output words.

### Model Evaluation

### Demo

## References

This repository is built based on the following resources:

[1] Quang-Vinh Dinh, AI VIET NAM. (2022). Transfer Learning (Text). AIO2022.

[2] PyTorch tutorial, [Language Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
