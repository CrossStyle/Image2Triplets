Image2Triplets
===========================

This is the implementation of Image2Tripplets, including a forward and backward model.

## Installation

```
conda env create -f requirement.yaml
```

## Preparation

1. Prepare the dataset. You can use gui.py in the hoi_dataset folder to annotate the HOI, but you should first annotate the objects and humans in the images in the format of the VOC dataset.
2. Prepare word embeddings. You can follow the instruction in the readme.md under the word_embeddings folder.

## Train

Backward process

```
python backward_model.py
```

Forward process

```
python forward_model.py
```

## Test

```
python demo.py
```

