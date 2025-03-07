# Deng (Dutch - English)

PyTorch implementation of a Transformer model for translation between English and Dutch.

## Features

- Bidirectional translation
- Transformer Architecture
- Tokenization
- Early Stop and Model Checkpoints
- CUDA support

## Project Structure

```
deng/
│
├── deng.py          # Main training module
├── test.py          # Testing and inference module
├── eng-dutch.tsv    # Training data
└── checkpoint/      # Saved model checkpoints
```

## Usage

### Training

To train the model:

```bash
python deng.py
```

The model will be saved automatically in the `checkpoint` directory when the validation loss improves.

### Testing

To test the trained model:

```bash
python test.py
```

## Dataset

The model is trained on the Tatoeba English-Dutch dataset stored in TSV format with the following columns:
- eng_id: English sentence ID
- eng: English sentence
- d_id: Dutch sentence ID
- dutch: Dutch sentence
