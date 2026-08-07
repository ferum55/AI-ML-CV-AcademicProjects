# PyTorch LSTM Music Generation

A deep learning project that trains a character-level LSTM network to generate music in ABC notation.

The project covers sequence data preparation, neural network training, loss optimization and autoregressive music generation using PyTorch.

## Model Architecture

The model is implemented as a custom `torch.nn.Module` with three main layers:

```text
Character Index
      ↓
Embedding
      ↓
LSTM
      ↓
Linear Layer
      ↓
Next-Character Probabilities
```

The network predicts the next character in an ABC music sequence based on the preceding sequence.

## Dataset Preparation

The music dataset is loaded using the MIT Introduction to Deep Learning utilities.

The preprocessing pipeline:

1. Loads songs represented in ABC notation.
2. Combines the songs into a single character sequence.
3. Builds a vocabulary of unique characters.
4. Maps characters to integer indices.
5. Creates input and target sequences for next-character prediction.
6. Groups sequences into training batches.

In the recorded notebook run, the dataset contained:

```text
817 songs
83 unique characters
```

## Training

The model is trained using:

- Cross-entropy loss
- Adam optimizer
- GPU acceleration with PyTorch
- Comet ML for experiment tracking
- Periodic model checkpoints

The main training configuration used in the notebook is:

```text
training iterations = 3000
batch size = 4
sequence length = 500
learning rate = 5e-3
embedding dimension = 256
hidden size = 256
```

The notebook tracks the loss during training and saves model checkpoints during the training process.

## Music Generation

After training, the model generates new music character by character.

The generation process:

1. Starts with an initial character.
2. Runs the current sequence through the LSTM.
3. Converts output logits into probabilities using softmax.
4. Samples the next character using multinomial sampling.
5. Feeds the generated character back into the model.
6. Repeats the process until the requested sequence length is reached.

The notebook generates a sequence of 1000 characters beginning with `X`, the standard beginning of an ABC tune definition.

In the recorded run, six song snippets were extracted from the generated sequence.

Generated ABC notation is also converted to audio for playback and can be saved as WAV files.

## Technologies

- Python
- PyTorch
- LSTM
- NumPy
- MIT Introduction to Deep Learning
- Comet ML
- SciPy
- ABC notation
- Google Colab

## Project File

[`PyTorch.ipynb`](./PyTorch.ipynb) contains the complete data preparation, model implementation, training loop and music generation workflow.

## Notes

This is an academic deep learning project focused on understanding recurrent neural networks, sequence modeling and autoregressive generation with PyTorch.
