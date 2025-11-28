import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# compare_predictions.py : This file contains scripts that compares          #
# the prediction results from the baseline FFNN model and the BiLSTM model.  #
# It computes various statistics and visualizations to analyze differences   #
# between the two models' outputs on the test dataset.                       #
# *Note*: This script assumes that both models have already made predictions #
# and saved them to 'submission_baseline.csv' and 'submission_bilstm.csv'.   #
##############################################################################

# Load the prediction files from both models
baseline = pd.read_csv("./model_predictions/baseline_predictions.csv")
bilstm = pd.read_csv("./model_predictions/advanced_predictions.csv")

# Drop the 'id' column for comparison since it's not needed
b = baseline.drop(columns = ["id"])
l = bilstm.drop(columns = ["id"])

# List of the 6 toxicity label names
labels = b.columns.tolist()

#--------------------Numerical/Statistical Comparisons---------------------#
# This section shows how confident each model is on average for each label
# Are BiLSTM predictions higher or lower on average?
print("\n==============================")
print("1. MEAN PREDICTIONS PER LABEL")
print("==============================")
mean_df = pd.DataFrame({
    "Baseline Mean": b.mean(),
    "BiLSTM Mean": l.mean(),
    "Difference": b.mean() - l.mean()
})
print(mean_df)

# This sections show the correlation between the two models' 
# predictions for each label
# Higher correlation means the models agree more often
print("\n==============================")
print("2. PEARSON CORRELATION (per label)")
print("==============================")
corr_df = b.corrwith(l)
print(corr_df)

# This section shows the average per-label difference in predictions
# How much do the models disagree on average?
print("\n==============================")
print("3. AVERAGE ABSOLUTE DIFF PER LABEL")
print("==============================")
abs_diff = (b - l).abs().mean()
print(abs_diff)

# This section shows the overall average absolute difference in predictions
# across all labels and examples
# How different are the models overall?
print("\n==============================")
print("4. GLOBAL AVERAGE ABSOLUTE DIFF")
print("==============================")
print((b - l).abs().values.mean())

#------------------------------Visualizations------------------------------#
print("\nGenerating visualizations...\n")
# HISTOGRAMS:
# This section creates histograms to visualize the distribution of 
# predictions from both models for each toxicity label.
# Shows whether models predict generally higher or lower probabilities
for label in labels:
    plt.figure(figsize = (8,5))
    plt.hist(baseline[label], bins = 50, alpha = 0.6, label="Baseline")
    plt.hist(bilstm[label], bins = 50, alpha = 0.6, label="BiLSTM")
    plt.title(f"Prediction Distribution for '{label}'")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()

    save_path = f"./images/histogram_{label}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram: {save_path}")


# SCATTER PLOTS:
# This section creates scatter plots to visualize the relationship
# between the predictions of the two models for each toxicity label.
# Each point is a test sample, showing how similarly or differently
# the models predicted its toxicity probability.
for label in labels:
    plt.figure(figsize = (6,6))
    plt.scatter(baseline[label], bilstm[label], alpha = 0.15, s = 5)
    plt.plot([0,1], [0,1], 'r--')  # diagonal
    plt.title(f"Scatter Plot: Baseline vs BiLSTM ({label})")
    plt.xlabel("Baseline")
    plt.ylabel("BiLSTM")
    
    save_path = f"./images/scatterplot_{label}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot: {save_path}")


# DIFFERENCE DISTRIBUTIONS:
# This section creates histograms to visualize the distribution of
# differences in predictions between the two models for each toxicity label.
# Shows whether one model tends to predict higher or lower probabilities.
for label in labels:
    diff = (baseline[label] - bilstm[label])
    plt.figure(figsize = (8,5))
    plt.hist(diff, bins = 50, alpha = 0.7)
    plt.title(f"Prediction Difference Distribution (Baseline - BiLSTM) for '{label}'")
    plt.xlabel("Difference in Probability")
    plt.ylabel("Frequency")

    save_path = f"./images/diff_dist_{label}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved difference distribution: {save_path}")


# CORRELATION HEATMAPS of Mean and Absolute Prediction Differences
# This section creates heatmaps to visualize the mean and average absolute
# differences in predictions between the two models across all toxicity labels.
# Shows overall patterns of agreement or disagreement.
plt.figure(figsize = (6,3))
sns.heatmap(mean_df, annot = True, cmap = "coolwarm", cbar = False)
plt.title("Mean Prediction Differences (Baseline - BiLSTM)")

plt.savefig("./images/heatmap_mean_difference.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved heatmap_mean_difference.png")


plt.figure(figsize = (6,3))
sns.heatmap(abs_diff.to_frame(), annot = True, cmap = "coolwarm", cbar = False)
plt.title("Average Absolute Prediction Differences")

plt.savefig("./images/heatmap_abs_difference.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved heatmap_abs_difference.png")