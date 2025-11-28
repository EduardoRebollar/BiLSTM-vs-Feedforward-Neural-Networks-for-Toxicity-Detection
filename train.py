import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import re
from utils import load_data, TextProcessor, convert_text_to_tensors
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

###########################################################################
# train.py : This file contains both of the models (Baseline FFNN and     #
# Advanced Bidirectional LSTM), the training loop and evaluation function #
# for multi-label toxicity classification                                 #
###########################################################################

#-------------------------------------------------------------#
# Determining the optimal max_length based on comment lengths #
#-------------------------------------------------------------#
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()

# Loads the train.csv to analyze comment lengths
df = pd.read_csv("./train/train.csv")

# Computes the lengths of comments
lengths = df["comment_text"].astype(str).apply(lambda x: len(tokenize(x)))

print("\nMedian length:", lengths.median())
print("Mean length:", lengths.mean())
print("75th percentile:", lengths.quantile(0.75))
print("85th percentile:", lengths.quantile(0.85))
print("Max length:", lengths.max())

#-------------------------------------------------------------------------------#
# Based on this analysis, I choose max_length = 75 for the models               #
# because it covers the majority of comments while keeping computations         #
# manageable since longer comments are less frequent improving on training time.#
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------#
# Baseline Model: Feedforward Neural Network (similar to PA2) #
#-------------------------------------------------------------#
class NeuralNetwork(nn.Module):
    '''
    A simple feedforward neural network for multi-label toxicity classification.
    Model architecture:
    - Embedding layer
    - Linear layer + ReLU
    - Dropout
    - Linear layer + ReLU
    - Dropout
    - Linear output layer
    
    *Note*: I use mean pooling over embeddings to reduce sequence length dimension.
    *Difference*: output_size = 6 and we will use BCEWithLogitsLoss
    for multi-label classification.
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, max_length = 75, dropout = 0.3):
        super(NeuralNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.dropout = dropout

        # The embedding layer that converts word indices to embeddings of a specified dimension
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Feed-forward NN layers with ReLU activation and dropout for regularization
        self.linear1 = nn.Linear(self.embedding_dim, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        '''
        Forward pass:
          x: A tensor with a shape of (batch_size, max_length)

        Returns:
          Output: Of shape (batch_size, output_size=6)
        '''
        embedded = self.embedding(x)                   # (batch_size, max_length, embedding_dim)
        linear1 = self.linear1(embedded)               # (batch_size, max_length, hidden_size)
        relu1 = self.relu(linear1)                     # (batch_size, max_length, hidden_size)
        collapsed = torch.mean(relu1, dim = 1)         # (batch_size, hidden_size)
        linear2 = self.linear2(collapsed)              # (batch_size, hidden_size)
        relu2 = self.relu2(linear2)                    # (batch_size, hidden_size)
        output = self.linear3(relu2)                   # (batch_size, output_size = 6)
        return output

#-----------------------------------------------#
# Advanced model: Bidirectional LSTM Classifier #
#-----------------------------------------------#
class AttentionBiLSTM(nn.Module):
    '''
    A bidirectional LSTM classifier for multi-label toxicity classification,
    with an attention mechanism to focus on important words/words that matter most.
    2 layers of BiLSTM captures more complex patterns and deeper insights in the text.
    Dropout is used for regularization to prevent overfitting.

    Model architecture:
    - Embedding layer
    - BiLSTM layer
    - Attention mechanism
    - Dropout
    - Fully connected output layer

    *Note*: output_size = 6 and we will use BCEWithLogitsLoss for multi-label classification.
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, max_length = 75, dropout = 0.4, num_layers = 2):
        super(AttentionBiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.dropout = dropout
        self.num_layers = num_layers

        # The embedding layer that converts word indices to embeddings of a specified dimension
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # A two layer BiLSTM with dropout between layers for regularization
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = num_layers, 
                            batch_first = True, bidirectional = True, dropout = dropout if num_layers > 1 else 0.0)

        # The attention layer to weigh LSTM outputs
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Fully connected layers for classification
        # This improves prediction stability and performance
        # (hidden * 2 because of output from both directions/bidirectional)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass:
          x: A tensor with a shape of (batch_size, max_length)

        Returns:
          Output: Of shape (batch_size, output_size=6)
        """
        # Embedding layer
        embedded = self.embedding(x)   # (batch, max_len, embedding_dim)

        # BiLSTM layer
        self.lstm_out, _ = self.lstm(embedded)  # (batch, max_len, hidden*2)

        # Compute attention-weighted BiLSTM outputs
        attention_weights = torch.softmax(self.attention(self.lstm_out), dim = 1)  # (batch, max_len)
        attention_outputs = torch.sum(attention_weights * self.lstm_out, dim = 1)  # (batch, hidden*2)

        # Classification layers
        fc1_out = self.fc1(attention_outputs)    # (batch, hidden_size)
        fc1_out = self.relu(fc1_out)             # (batch, hidden_size)
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)               # (batch, output_size = 6)
        return output

#----------------------------------------#
# Training loop (shared for both models) #
#----------------------------------------#
def train(model, train_features, train_labels, test_features, test_labels,
          num_epochs = 25, learning_rate = 0.001, batch_size = 64):
    """
    Train the model (baseline or BiLSTM) for multi-label toxicity classification.

    Args:
        model: The neural network model
        train_features: training features represented by token indices (tensor)
        train_labels: train labels tensor of shape (N, 6), float
        test_features: test features represented by token indices (tensor) (unused in training loop)
        test_labels: test labels tensor (unused in training loop)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size 

    Returns:
        Trained model

    Training loop:
       - Using Adam optimizer with weight decay for regularization
       - Using BCEWithLogitsLoss for multi-label outputs
       - Using pos_weight for imbalance handling since toxic comments are rare
       so the model learns to upweight rare positive labels and not just predict all zeros.
    """

    # Calculate positive weights for each label to handle class imbalance
    # pos_weight = num_neg / num_pos
    num_samples = train_labels.shape[0]
    num_pos = train_labels.sum(dim = 0)
    num_neg = num_samples - num_pos
    pos_weights = num_neg / (num_pos + 1e-6)  # avoid division by zero
    print(f"Pos weights for each label: {pos_weights}")

    # Loss function for multi-label classification with class imbalance handling
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weights)

    # Adam optimizer with weight decay for regularization helping prevent overfitting
    # It is common practice to use the same value for the weight decay as the learning rate
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = learning_rate)

    # Wrap training tensors into a PyTorch DataLoader
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Iterate over mini-batches
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()   # Zero the gradients from the previous step

            outputs = model(inputs)           # (batch_size, 6)
            loss = loss_fn(outputs, labels)   # labels are float (0/1) with a shape of (batch_size,6)

            loss.backward()     # Back propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to prevent exploding gradients
            optimizer.step()    # Update weights

            total_loss += loss.item()   # Accumulate the loss

        avg_loss = total_loss / len(train_dataloader)   # Average loss for the epoch
        print(f"Epoch {epoch}: average training loss: {avg_loss:.4f}")

    return model
    
# ---------------------------------------------#
# Evaluation function (shared for both models) #
# ---------------------------------------------#
def evaluate(model, test_features, test_labels):
    """
    Evaluate the trained model on validation/test data for 
    multi-label toxicity classification.

    Args:
        model: The trained neural network model
        test_features: (tensor) shape (N, max_length)
        test_labels: (tensor) shape (N, 6) with 0/1 floats

    Returns a dictionary of evaluation metrics including:
        - Subset accuracy
        - Macro & micro precision/recall/F1
        - Per-label precision/recall/F1
        - Per-label confusion matrices (TP, FP, FN, TN)
    """
    #######################
    # Evaluation:
    #  - Use torch.no_grad()
    #  - Applying sigmoid to outputs since BCEWithLogitsLoss was used during training
    #    and we need probabilities in [0,1].
    #  - Threshold at 0.3 to get binary predictions per label
    #  - (Because 0.5 tends to predict all zeros due to class imbalance)
    #  - (Hopefully 0.3 encourages more positive predictions)
    #######################


    # Define label_names here properly
    label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # Model set to evaluation mode
    model.eval()

    with torch.no_grad():   # Disable gradient calculations for efficiency
        outputs = model(test_features)  # Shape of (N, 6)
        probs = torch.sigmoid(outputs)  # Shape of (N, 6) probabilities in [0,1]
        preds = (probs >= 0.3).int()    # Shape of (N, 6) binary predictions with threshold 0.3
        np_labels = test_labels.numpy() # Convert labels to numpy
        np_preds = preds.numpy()        # Convert predictions to numpy

        #-----------------------------#
        #       Global Metrics        #
        #-----------------------------#
        subset_acc = accuracy_score(np_labels, np_preds)    # subset accuracy: all 6 labels must match to count as correct

        macro_p = precision_score(np_labels, np_preds, average = "macro", zero_division = 0)    # macro precision
        macro_r = recall_score(np_labels, np_preds, average = "macro", zero_division = 0)   # macro recall
        macro_f1 = f1_score(np_labels, np_preds, average = "macro", zero_division = 0)  # macro F1 score

        micro_p = precision_score(np_labels, np_preds, average = "micro", zero_division = 0)    # micro precision
        micro_r = recall_score(np_labels, np_preds, average = "micro", zero_division = 0)   # micro recall
        micro_f1 = f1_score(np_labels, np_preds, average = "micro", zero_division = 0)  # micro F1 score

        #-----------------------------#
        #      Per-label metrics      #
        #-----------------------------#
        per_label_precision = precision_score(np_labels, np_preds, average = None, zero_division = 0)   # per-label precision
        per_label_recall = recall_score(np_labels, np_preds, average = None, zero_division = 0)      # per-label recall
        per_label_f1 = f1_score(np_labels, np_preds, average = None, zero_division = 0)           # per-label F1 score

        print("\n===== PER-LABEL METRICS =====")
        print("{:<15} {:<10} {:<10} {:<10}".format("Label", "Precision", "Recall", "F1"))   # Header
        for i, name in enumerate(label_names):                                            # Rows
            print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f}".format(                        # Data
                name, per_label_precision[i], per_label_recall[i], per_label_f1[i]
            ))

        #------------------------------#
        # Per-label confusion matrices #
        #------------------------------#
        print("\n===== PER-LABEL CONFUSION MATRICES =====")
        conf_matrices = {}      # Dictionary to hold confusion matrix for each label

        # Calculate TP, FP, FN, TN for each label
        for i, name in enumerate(label_names):
            tp = np.sum((np_labels[:, i] == 1) & (np_preds[:, i] == 1))
            fp = np.sum((np_labels[:, i] == 0) & (np_preds[:, i] == 1))
            fn = np.sum((np_labels[:, i] == 1) & (np_preds[:, i] == 0))
            tn = np.sum((np_labels[:, i] == 0) & (np_preds[:, i] == 0))

            # Store matrix in dictionary
            conf_matrices[name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

            print(f"\nLabel: {name}")
            print(f"TP: {tp} | FP: {fp}")
            print(f"FN: {fn} | TN: {tn}")

        #----------------------------------#
        # Return results in a clean format #
        #----------------------------------#
        return {
            "subset_accuracy": subset_acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "per_label_precision": per_label_precision,
            "per_label_recall": per_label_recall,
            "per_label_f1": per_label_f1,
            "confusion_matrices": conf_matrices,
        }

#--------------------------------------------------------------------#
#                          Main function                             #
#--------------------------------------------------------------------#
if __name__ == "__main__":
    '''
    Main script adapted for final project:
      - Loads train.csv
      - Multi-label (6 labels)
      - Uses same TextProcessor pipeline as PA2
      - Splits data into training and validation sets (25% train, 5% test)
      - Preprocesses text and builds vocabulary from training data
      - Converts text to tensor representations of word indices
      - Trains both the Baseline Feedforward NN and the Advanced BiLSTM model
      - Evaluates both models on the validation set and prints results
    '''   
    # Load all labeled data from train.csv
    train_csv_file = './train/train.csv'
    all_texts, all_labels = load_data(train_csv_file)  # labels: (N, 6) float tensor

    # Ensure labels are present if not raise an error
    if all_labels is None:
        raise ValueError("Training CSV must contain toxicity label columns.")

    # Convert labels to numpy for splitting
    all_labels_np = all_labels.numpy()

    # Split into training and validation sets (25% train and 5% test, ends up being a ratio of 5:1)
    # Note: multi-label stratification is more complex, so I use a simple random split here.
    train_texts, test_texts, train_labels_np, test_labels_np = train_test_split(
        all_texts, all_labels_np, test_size = 0.05, train_size = 0.25, random_state = 42)

    # Convert labels back to tensors
    train_labels = torch.tensor(train_labels_np, dtype = torch.float32)
    test_labels = torch.tensor(test_labels_np, dtype = torch.float32)

    # Preprocess text and build vocabulary from training texts
    processor = TextProcessor(vocab_size = 20000)
    processor.build_vocab(train_texts)

    # Convert text documents to tensor representations of word indices
    max_length = 75 # Must match the model's expected input length
    train_features = convert_text_to_tensors(train_texts, processor, max_length)    # (N_train, max_length)
    test_features = convert_text_to_tensors(test_texts, processor, max_length)      # (N_test, max_length)

    # Create a neural network model 
    vocab_size = len(processor.word_to_idx)
    embedding_dim = 100
    output_size = 6  # Multi-label: 6 toxicity categories

    # Baseline model hyperparameters
    base_hidden_size = 64
    base_dropout = 0.3

    # BiLSTM model hyperparameters
    bilstm_hidden_size = 128
    bilstm_dropout = 0.4

    #---------------------------------------------------------------#
    # Data verification summary
    print("\n")
    print("Data verification summary:")
    print("-" * 27)
    print(f"Train size: {len(train_features)} samples")
    print(f"Test size:  {len(test_features)} samples")
    print(f"Train tensor shape: {train_features.shape}, Labels shape: {train_labels.shape}")
    print(f"Test tensor shape:  {test_features.shape}, Labels shape: {test_labels.shape}")
    print("-" * 50)
    #---------------------------------------------------------------#

    
    #---------------------------------------------------------------#
    #          Training the Baseline Neural Network Model           #
    #---------------------------------------------------------------#
    print("\n")
    print("Training the Baseline Feedforward Neural Network Model")
    print("-" * 50)

    # Create the baseline NN model
    base_model = NeuralNetwork(vocab_size, embedding_dim, base_hidden_size, output_size, max_length, base_dropout)

    # Train the baseline NN model
    trained_base = train(base_model, train_features, train_labels, test_features, test_labels,
                          num_epochs = 25, learning_rate = 0.001, batch_size = 64)

    # Evaluate the baseline NN model
    base_results = evaluate(trained_base, test_features, test_labels)

    # Evaluation results
    print("\n")
    print(f"Baseline Model Performance (multi-label macro averages):")
    print("---------------------------------------------------------")
    print(f"Subset Accuracy: {base_results['subset_accuracy']:.4f}")
    print(f"Macro Precision: {base_results['macro_precision']:.4f}")
    print(f"Macro Recall:    {base_results['macro_recall']:.4f}")
    print(f"Macro F1 score:  {base_results['macro_f1']:.4f}")

    # Saving baseline model to file
    outfile = './trained_model_paths/trained_baseline_model.pth'
    torch.save(trained_base.state_dict(), outfile)
    print(f"\nTrained baseline model saved to {outfile}\n")
    #---------------------------------------------------------------#
    
    #---------------------------------------------------------------#
    #       Training the Advanced Bidirectional LSTM Model          #
    #---------------------------------------------------------------#
    print("\n===== Training BiLSTM Model =====\n")
    advanced_model = AttentionBiLSTM(vocab_size, embedding_dim, bilstm_hidden_size, output_size, max_length, bilstm_dropout, num_layers = 2)

    # Train the advanced BiLSTM model
    trained_advanced = train(advanced_model, train_features, train_labels, test_features, test_labels,
                             num_epochs = 25, learning_rate = 0.001, batch_size = 64)

    # Evaluate the advanced model
    advanced_results = evaluate(trained_advanced, test_features, test_labels)

    print("\nAdvanced Model Performance (macro averages):")
    print("-----------------------------------------------")
    print(f"Subset Accuracy: {advanced_results['subset_accuracy']:.4f}")
    print(f"Macro Precision: {advanced_results['macro_precision']:.4f}")
    print(f"Macro Recall:    {advanced_results['macro_recall']:.4f}")
    print(f"Macro F1 score:  {advanced_results['macro_f1']:.4f}")

    # Save advanced model
    outfile2 = './trained_model_paths/trained_advanced_model.pth'
    torch.save(trained_advanced.state_dict(), outfile2)
    print(f"\nAdvanced model saved to {outfile2}\n")
    #----------------------------------------------------------------#

    #------------------- Side-by-side comparison --------------------#
    print("\n================ MODEL COMPARISON ================")
    print("Metric                 | Feedforward | Advanced BiLSTM")
    print("-------------------------------------------------")
    print(f"Subset Accuracy        | {base_results['subset_accuracy']:.4f}      | {advanced_results['subset_accuracy']:.4f}")
    print(f"Macro Precision        | {base_results['macro_precision']:.4f}      | {advanced_results['macro_precision']:.4f}")
    print(f"Macro Recall           | {base_results['macro_recall']:.4f}      | {advanced_results['macro_recall']:.4f}")
    print(f"Macro F1 Score         | {base_results['macro_f1']:.4f}      | {advanced_results['macro_f1']:.4f}")
    print("=================================================\n")