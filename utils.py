import torch
import re
import pandas as pd
from collections import Counter

#-----------------------------------------------------------------------------------------#
# utils.py : This file contains utilities for loading the datasets and text preprocessing.#
# The TextProcessor class contains methods similar to those in PA2                        #
# These functions are used by both the baseline FFNN model and the BiLSTM model.          #
#-----------------------------------------------------------------------------------------#

def load_data(infile):
    """
    Loads the texts and labels from a CSV file in the following format:

        id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

    Returns:
        texts: list of comment strings
        labels: a tensor with a shape of (N, 6) or None (if no labels are present)

        *Note*: Pytorch requires float tensors for multi-label classification due
        to the use of sigmoid activation and BCEWithLogitsLoss.  
    """
    # CSV mode (for final project instead of PA2's txt files)
    if infile.lower().endswith(".csv"):

        # Load CSV file
        df = pd.read_csv(infile)
        if "comment_text" not in df.columns:
            raise ValueError(f"CSV file {infile} must contain a 'comment_text' column.")

        # Extract comment texts
        texts = df["comment_text"].astype(str).tolist()

        # The 6 toxicity label columns
        label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        # If all toxicity label columns exist, return multi-label tensor
        if all(col in df.columns for col in label_cols):
            labels = df[label_cols].values
            labels = torch.tensor(labels, dtype=torch.float32)   # Converts labels to a float tensor of shape (N, 6)

        else:
            # For test.csv where labels are not present
            labels = None

        return texts, labels

def load_test_data(infile):
    """
    Loads the test data from a test.csv file in the following format:
        id, comment_text

    Returns:
        ids: as a list of row IDs as strings
        texts: list of comment strings
    """
    # Load CSV file
    df = pd.read_csv(infile)

    # If both "id" and "comment_text" columns exist, extract and return them, if not raise an error
    if "id" not in df.columns or "comment_text" not in df.columns:
        raise ValueError(f"Test CSV {infile} must contain 'id' and 'comment_text' columns.")
    ids = df["id"].astype(str).tolist()
    texts = df["comment_text"].astype(str).tolist()
    return ids, texts

class TextProcessor:
    """
    Preprocess text, build vocabulary and mappings between words and indices
    (same structure as PA2, reused for final project)
    """
    def __init__(self, vocab_size = 20000):
        # We restrict vocabulary size to a pre-defined value to save memory
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}

    def tokenize(self, text):
        # Simple tokenization, remove punctuation and convert to lowercase
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def build_vocab(self, texts):
        # Build vocabulary, mappings between words and indices from training corpus
        word_counts = Counter()

        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)

        # Build vocabulary based on the most common words, constrained by vocabulary size
        # Reserve two spaces for <PAD> and <UNK>
        most_common = word_counts.most_common(self.vocab_size - 2)

        # Create word to index mapping
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

        # Add remaining words to the mappings
        for i, (word, count) in enumerate(most_common):
            self.word_to_idx[word] = i + 2
            self.idx_to_word[i + 2] = word

    def pad_sequence(self, sequence, max_length = 25):
        # Truncate a sequence if too long (> max_length), pad if too short (< max_length)
        if len(sequence) < max_length:
            return sequence + ["<PAD>"] * (max_length - len(sequence))
        return sequence[:max_length]

    def text_to_indices(self, text, max_length = 25):
        # Convert text to sequence of word indices.
        # Pad/Truncate the sequence to max_length.
        words = self.tokenize(text)
        words = self.pad_sequence(words, max_length)

        # Convert to indices after tokenization and padding
        indices = [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in words]

        return torch.tensor(indices, dtype = torch.long)

    def get_vocab_size(self):
        # Return the vocabulary size
        return len(self.word_to_idx)

    def get_word(self, idx):
        # Access word by index
        return self.idx_to_word.get(idx, '<UNK>')

    def get_idx(self, word):
        # Access index by word
        return self.word_to_idx.get(word, self.word_to_idx['<UNK>'])

def convert_text_to_tensors(docs, processor, max_length = 100):
    """
    This function prepares training features given a text corpus and a TextProcessor instance.
    It converts raw texts into tensor representations of word indices.

    Args:
        docs: List of raw text strings
        processor: a TextProcessor instance with built vocabulary
        max_length: Maximum sequence length

    Returns:
        Tensor of shape (num_texts, max_length)
    """
    token_indices = []

    for i, text in enumerate(docs):
        indices = processor.text_to_indices(text, max_length)
        token_indices.append(indices)

    token_indices = torch.stack(token_indices)

    return token_indices