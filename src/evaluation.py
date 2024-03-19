import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES_FILE = "../archive/TrainSet.csv"
EVALUATIONS_PATH = "../results"
PLOT_PATH = "../plots"
PLOT_SIZE = (16, 6)
TOP_FEATURES = 5


def get_label_names(indexes):
    """Return the names of the selected features."""
    with open(FEATURES_FILE, "r") as file:
        label_names = file.readline().strip().split(",")[1:]
    return [label_names[i] for i in indexes]


def get_evaluation_files():
    """Return all evaluation results files."""
    return [file for file in os.listdir(EVALUATIONS_PATH) if file.endswith('.json')]


def plot_feature_counts(feature_counts, title, ax):
    """Plot the feature counts on the given axes."""
    sns.barplot(x=feature_counts.index, y=feature_counts.values, ax=ax)
    ax.set_xlabel('Selected Features')
    ax.set_ylabel('Count')
    ax.set_title(title)


def plot_total_feature_counts(feature_counts):
    """Plot the total count of each feature."""
    feature_counts = feature_counts.groupby(feature_counts.index).sum()
    feature_counts = feature_counts.sort_values(ascending=False)

    plt.figure(figsize=PLOT_SIZE)
    sns.barplot(x=feature_counts.index, y=feature_counts.values)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    for i, bar in enumerate(plt.gca().patches):
        color = 'teal' if i < TOP_FEATURES else 'grey'
        bar.set_color(color)
        if i < TOP_FEATURES:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     round(bar.get_height(), 2), ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

    plt.xlabel('Selected Features')
    plt.ylabel('Count')
    plt.title('Total Count of Each Selected Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "feature_counts_total.png"))
    plt.show()


def main():
    """ Main function to plot the feature counts."""
    file_names = get_evaluation_files()
    fig, axes = plt.subplots(len(file_names), figsize=(PLOT_SIZE[0], PLOT_SIZE[1] * len(file_names)))
    plt.title('Count of Each Selected Feature')

    feature_counts_total = []
    for i, file_name in enumerate(file_names):
        df = pd.read_json(f'{EVALUATIONS_PATH}/{file_name}')
        flattened_features = [item for sublist in df['selected_features'] for item in sublist]
        feature_names = get_label_names(flattened_features)
        feature_names = [f"{name} \n{index}" for name, index in zip(feature_names, flattened_features)]
        feature_counts = pd.Series(feature_names).value_counts()
        feature_counts_total.append(feature_counts)
        plot_feature_counts(feature_counts, f'Count of Selected Features for {file_name}', axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "feature_counts_per_subset.png"))
    plt.show()

    feature_counts_total = pd.concat(feature_counts_total)
    plot_total_feature_counts(feature_counts_total)


if __name__ == "__main__":
    main()