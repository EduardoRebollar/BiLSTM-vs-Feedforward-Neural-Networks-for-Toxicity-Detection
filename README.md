# BiLSTM-vs-Feedforward-Neural-Networks-for-Toxicity-Detection
**Author(s):** Eduardo Rebollar

Toxic online interactions often contain language such as insults, threats, and hate speech, which can lead to individuals feeling harassed and being socially excluded. For this reason, social media platforms aim to develop fair, transparent, and accurate models that contribute to safer online spaces and more inclusive communities. Because moderating content manually is very expensive and can even introduce individual biases from each moderator, these platforms have increasingly relied on automated tools that can detect toxicity and abusive content. 

This project focuses on building two distinct, but similar models that classify user comments into six toxicity categories `(toxic, severe_toxic, obscene, threat, insult, identity_hate)` using the Jigsaw Toxic Comment Classification dataset. The goal of this project is to design, train, and evaluate a baseline FeedForward Neural Network (FFNN) against an improved advanced Bidirectional LSTM (BiLSTM) with attention and compare their performance using standard multi-label evaluation metrics. The project includes preprocessing tools, training scripts, prediction scripts, and a comparison tool to analyze differences between the models.

## Project/Repository Structure

```
BiLSTM vs Feedforward Neural Networks for Toxicity Detection/
│
├── train/
│   └── train.csv                      #  Main training dataset from Jigsaw (comments + 6 toxicity labels) 
│
├── test/
│   └── test.csv                       #  Unlabeled Jigsaw test set used for predictions
│
├── trained_model_paths/
│   ├── trained_baseline_model.pth     #  Saved FFNN model weights
│   └── trained_advanced_model.pth     #  Saved BiLSTM model weights
│
├── model_predictions/
│   ├── baseline_predictions.csv       #  Full prediction probabilities (6 toxicity labels) from the FFNN model
│   └── advanced_predictions.csv       #  Full prediction probabilities (6 toxicity labels) from the BiLSTM model
│
├── images/
│   ├── histogram_toxic.png            #  Histogram of toxic prediction probabilities (FFNN vs BiLSTM)
│   ├── histogram_severe_toxic.png     #  Histogram for severe_toxic predictions
│   ├── histogram_obscene.png          #  Histogram for obscene predictions
│   ├── histogram_threat.png           #  Histogram for threat predictions
│   ├── histogram_insult.png           #  Histogram for insult predictions
│   ├── histogram_identity_hate.png    #  Histogram for identity_hate predictions
│   │
│   ├── scatterplot_toxic.png          #  Scatter plot FFNN vs BiLSTM on toxic label
│   ├── scatterplot_severe_toxic.png   #  Scatter plot for severe_toxic predictions
│   ├── scatterplot_obscene.png        #  Scatter plot for obscene predictions
│   ├── scatterplot_threat.png         #  Scatter plot for threat predictions
│   ├── scatterplot_insult.png         #  Scatter plot for insult predictions
│   ├── scatterplot_identity_hate.png  #  Scatter plot for identity_hate predictions
│   │
│   ├── diff_dist_toxic.png            #  Distribution of (FFNN - BiLSTM) differences for toxic
│   ├── diff_dist_severe_toxic.png     #  Difference distribution for severe_toxic
│   ├── diff_dist_obscene.png          #  Difference distribution for obscene
│   ├── diff_dist_threat.png           #  Difference distribution for threat
│   ├── diff_dist_insult.png           #  Difference distribution for insult
│   ├── diff_dist_identity_hate.png    #  Difference distribution for identity_hate
│   │ 
│   ├── heatmap_mean_difference.png    #  Heatmap comparing mean probability differences between models
│   └── heatmap_abs_difference.png     #  Heatmap comparing average absolute differences between models
│
├── utils.py                           #  Preprocessing, vocabulary building, and data loading utilities
├── train.py                           #  Trains both FFNN and BiLSTM models
├── predict.py                         #  Generates predictions for test.csv (both models)
├── compare_predictions.py             #  Compares baseline vs BiLSTM predictions w/ visualizations
│
├── final_project_report.pdf           #  Detailed explanations and expanded analysis of the project
├── requirements.txt                   #  Project dependencies
├── LICENSE                            #  Project license
└── README.md                          #  Project documentation
```

## How to Reproduce / Installation
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run scripts in order:**
   ```bash
   python train.py                  #  Trains both the FFNN and BiLSTM models, prints evaluation metrics, and saves model weights
   python predict.py                #  Generates predictions for test.csv (both models)
   python compare_predictions.py    #  Compares FFNN vs BiLSTM predictions w/ statistics & visualizations
   ```
3. **View results (generated after running scripts):**
   - Model Weights: `trained_model_paths/` folder
   - Predictions: `model_predictions/` folder
   - Visualizations: `images/` folder

## Optional: Cleanup Pre-Generated Outputs Before Training Your Own Model
If you want to train your own fresh models instead of using the pre-generated artifacts in this repository, you may delete what is found inside the following folders:
```bash
trained_model_paths/      # Contains pretrained baseline & BiLSTM .pth model files
model_predictions/        # Contains prediction CSVs produced by predict.py
images/                   # Contains all visualizations generated by compare_predictions.py
```
Removing these folders ensures that your new training run produces completely fresh model checkpoints, predictions, and plots.

---

## References

### Full Project Report
A comprehensive written report that delves into the project's background, methodology, dataset description, model architectures, hyperparameters, experimental results, discussion, and future work that includes detailed explanations and expanded analysis is available in the repository.

You can find the full report here:

`/final_project_report.pdf`

### Data Sources
This project uses the Jigsaw Toxic Comment Classification Challenge dataset, released by Jigsaw/Google as part of a Kaggle competition.
The dataset consists of user comments from Wikipedia Talk Pages, labeled for toxicity by human annotators.

cjadams, Jeffrey Sorensen, Julia Elliott, Lucas Dixon, Mark McDonald, nithum, and Will Cukierski. 2017. Toxic Comment Classification Challenge. https://kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge, Kaggle.

### Libraries
- matplotlib, numpy, pandas, scikit-learn, seaborn, torch

---
