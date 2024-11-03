import pandas as pd
from sklearn.metrics import f1_score


def load_data(submission_file, labels_file):
    """Load submission and true labels data."""
    submission_df = pd.read_csv(submission_file, index_col='uid')
    labels_df = pd.read_csv(labels_file, index_col='uid')
    return submission_df, labels_df


def calculate_f1_scores(submission_df, labels_df):
    """Calculate F1 score for each column (target variable) and return the average."""
    scores = []

    # Iterate over each target variable in the labels
    for col in labels_df.columns:
        y_true = labels_df[col]
        y_pred = submission_df[col]

        # Calculate F1 score for binary or categorical data
        if len(labels_df[col].unique()) == 2:  # Binary variable
            score = f1_score(y_true, y_pred, average='binary')
        else:  # Categorical variable
            score = f1_score(y_true, y_pred, average='micro')

        scores.append(score)
        print(f"F1 score for {col}: {score * 100:.2f}%")

    # Calculate and return average F1 score across all columns
    avg_score = sum(scores) / len(scores)
    print(f"Average F1 score: {avg_score * 100:.2f}%")
    return avg_score


def main():
    submission_file = 'submission.csv'
    labels_file = 'data/test_labels.csv'

    # Load submission and labels data
    print("Loading data...")
    submission_df, labels_df = load_data(submission_file, labels_file)

    # # Ensure both dataframes have the same columns and uids
    # assert submission_df.shape == labels_df.shape, "Shape mismatch between submission and labels."
    # assert all(submission_df.columns == labels_df.columns), "Column mismatch between submission and labels."
    # assert all(submission_df.index == labels_df.index), "UID mismatch between submission and labels."

    # Calculate F1 scores
    print("Calculating F1 scores...")
    calculate_f1_scores(submission_df, labels_df)


if __name__ == "__main__":
    main()
