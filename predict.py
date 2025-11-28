import torch
import numpy as np
import pandas as pd
from utils import load_data, load_test_data, TextProcessor, convert_text_to_tensors
from train import AttentionBiLSTM, NeuralNetwork

#########################################################################################
# utils.py : This file contains functions that loads both pretrained models,            #
# makes predictions on test.csv data, and saves the results to two different CSV files. #
# Supports: Baseline FFNN model + BiLSTM model                                          #
# *Note*: - Hyperparameters used here must match those used during training in train.py #
#           (e.g., vocab size, embedding dim, hidden size, dropout, max length, etc.)   #
#         - Both models must be trained and saved prior to running this script.         #
#########################################################################################

def load_model(model_name, model_path, vocab_size, embedding_dim, hidden_size, output_size, max_length, dropout):
    """
    Load a pre-trained multi-label toxicity model (baseline FFNN or BiLSTM) from file.
    
    Args:
        model_name: Name of the model ("FFNN" or "BiLSTM")
        model_path: Path to the saved .pth model file
        vocab_size: Size of the vocabulary used in training
        embedding_dim: Dimension of word embedding vectors
        hidden_size: Size of hidden layer (differs for each model)
        output_size: Number of output labels (6 for toxicity types)
        max_length: Maximum sequence length used during training
        dropout: Dropout rate for regularization (differs for each model)
    
    Returns:
        A loaded model in evaluation mode
    """
    # Creates an empty model instance
    model = model_name(vocab_size, embedding_dim, hidden_size, output_size, max_length, dropout)

    # Loads the stored model weights
    model.load_state_dict(torch.load(model_path))

    # Sets the model to evaluation mode, since certain layers act differently during training and evaluation
    model.eval()

    print(f"Loaded trained {model_name} model from: {model_path}")

    return model

def predict_test_data(model, processor, test_ids, test_texts, outfile,
                      batch_size = 1000, max_length = 75):
    """
    Make predictions for test.csv comments and saves a predictions CSV file.

    Args:
        model: Pre-trained neural network model for toxicity classification
        processor: a TextProcessor instance (same vocab as training)
        test_ids: list of IDs from test.csv
        test_texts: list of comment_text strings from test.csv
        outfile: Path to save CSV predictions
        batch_size: Batch size for processing
        max_length: Maximum sequence length for padding/truncating
    """

    # List to store probabilities for all examples
    all_probs = []

    # Model is in evaluation mode
    model.eval()

    with torch.no_grad():   # This disables gradient calculations for efficiency
        # Process texts in batches
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i + batch_size]

            # Convert batch texts to tensors of padded/truncated word indices
            batch_tensors = convert_text_to_tensors(batch_texts, processor, max_length)

            # Forward pass through the model
            outputs = model(batch_tensors)  # Produces raw outputs
            probs = torch.sigmoid(outputs)  # Converts outputs to probabilities

            # Converts the batch probabilities from tensors to numpy and appends to list
            all_probs.append(probs.numpy())

    # Concatenates all batches into a single array of shape (N, 6)
    all_probs = np.vstack(all_probs)

    # Creates an output dataframe
    df_out = pd.DataFrame(all_probs, columns = [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    df_out.insert(0, "id", test_ids)    # Insert 'id' column at the front

    # Saves the predictions to a CSV file
    df_out.to_csv(outfile, index = False)
    print(f"Saved predictions for ({len(test_ids)} rows) comments to '{outfile}'")

if __name__ == "__main__":
    # Load data files 
    train_file = './train/train.csv'
    test_file = './test/test.csv'

    print("\nPredicting toxicity probabilities for test data...\n")

    # Load training data to build vocabulary (required for text preprocessing)
    train_texts, train_labels = load_data(train_file)
    if train_labels is None:
        raise ValueError("Training CSV must contain label columns for building the model.")

    # Load test data (ids and texts)
    test_ids, test_texts = load_test_data(test_file)

    #######################################################
    # Hyperparameters that matches those used in train.py #
    #######################################################

    # Defining model parameters
    vocab_size_limit = 20000
    max_length = 75
    embedding_dim = 100
    output_size = 6  # six toxicity categories

    # Baseline model hyperparameters
    base_hidden_size = 64
    base_dropout = 0.3

    # BiLSTM model hyperparameters
    bilstm_hidden_size = 128
    bilstm_dropout = 0.4

    # Build text processor and vocabulary from training data
    processor = TextProcessor(vocab_size = vocab_size_limit)
    processor.build_vocab(train_texts)
    vocab_size = len(processor.word_to_idx)
    print(f"Vocabulary size: {vocab_size} words (limit was {vocab_size_limit})\n")

    # Load the trained models
    base_model_path = "./trained_model_paths/trained_baseline_model.pth"
    advanced_model_path = "./trained_model_paths/trained_advanced_model.pth"

    # Load Baseline FFNN model
    base_model = load_model(NeuralNetwork, base_model_path, vocab_size, embedding_dim,
                       base_hidden_size, output_size, max_length, base_dropout)
    
    # Load Advanced BiLSTM model
    advanced_model = load_model(AttentionBiLSTM, advanced_model_path, vocab_size, embedding_dim,
                       bilstm_hidden_size, output_size, max_length, bilstm_dropout)


    # Predict probabilities for test.csv using both models and saves to separate CSV files

    print("\nRunning Baseline model prediction...")
    predict_test_data(base_model, processor, test_ids, test_texts, outfile = "./model_predictions/baseline_predictions.csv", max_length = max_length)

    print("\nRunning BiLSTM model prediction...")
    predict_test_data(advanced_model, processor, test_ids, test_texts, outfile = "./model_predictions/advanced_predictions.csv", max_length = max_length)

    print("\nAll predictions are complete!")