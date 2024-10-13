import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import re

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters,
    tokenizing, and lemmatizing.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text  # No need for tokenization or stop words removal now

def load_data(features_file, labels_file=None):
    """Load feature and label data from CSV files."""
    features_df = pd.read_csv(features_file)
    if labels_file is not None:
        labels_df = pd.read_csv(labels_file)
        return features_df, labels_df
    else:
        return features_df, None

def prepare_data(features_df, labels_df):
    """Prepare data by combining narratives and preprocessing."""
    features_df['processed_narrative'] = (features_df['NarrativeLE'].fillna('') + ' ' +
                                          features_df['NarrativeCME'].fillna(''))
    features_df['processed_narrative'] = features_df['processed_narrative'].apply(preprocess_text)

    # Merge features and labels
    merged_df = pd.merge(features_df[['uid', 'processed_narrative']], labels_df, on='uid', how='inner')
    return merged_df

def create_tfidf_features(texts, max_features=10000):
    """Create TF-IDF features from preprocessed texts using CountVectorizer."""
    # vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')  # Using built-in stop words
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))  # Using bigrams
    return vectorizer.fit_transform(texts), vectorizer

def train_and_evaluate_model(X, y):
    """Train a model for each label and evaluate using cross-validation."""
    models = []
    scores = []

    for col in tqdm(y.columns, desc="Training models"):
        # model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        # score = cross_val_score(model, X, y[col], cv=5, scoring='f1_micro').mean()
        score = cross_val_score(model, X, y[col], cv=3, scoring='f1_micro').mean()

        model.fit(X, y[col])  # Fit the model on all data
        models.append(model)
        scores.append(score)
        print(f"F1 score for {col}: {score * 100:.2f}%")

    avg_score = np.mean(scores)
    print(f"Average F1 score across all labels: {avg_score * 100:.2f}%")
    return models, avg_score

def make_predictions(models, X):
    """Make predictions using the trained models."""
    predictions = []
    for model in models:
        predictions.append(model.predict(X))
    return np.column_stack(predictions)

def save_submission(predictions, uids, columns, output_file):
    """Save predictions to a submission file with UID as index."""
    submission_df = pd.DataFrame(predictions, columns=columns)
    submission_df['uid'] = uids
    submission_df.set_index('uid', inplace=True)
    submission_df.to_csv(output_file)
    print(f"Submission saved to {output_file}")

def main():
    # File paths
    train_features_file = 'assets/train_features.csv'
    train_labels_file = 'assets/train_labels.csv'
    test_features_file = 'data/test_features.csv'
    submission_format_file = 'data/submission_format.csv'
    output_submission_file = 'submission.csv'

    # Load and prepare training data
    print("Loading and preparing training data...")
    train_features_df, train_labels_df = load_data(train_features_file, train_labels_file)
    train_df = prepare_data(train_features_df, train_labels_df)

    # Create TF-IDF features
    print("Creating TF-IDF features...")
    X_train, vectorizer = create_tfidf_features(train_df['processed_narrative'])
    y_train = train_df.drop(['uid', 'processed_narrative'], axis=1)

    # Train and evaluate models
    print("Training and evaluating models...")
    models, avg_score = train_and_evaluate_model(X_train, y_train)

    # Load and prepare test data
    print("Loading and preparing test data...")
    test_features_df, _ = load_data(test_features_file, None)
    test_features_df['processed_narrative'] = (test_features_df['NarrativeLE'].fillna('') + ' ' +
                                               test_features_df['NarrativeCME'].fillna(''))
    test_features_df['processed_narrative'] = test_features_df['processed_narrative'].apply(preprocess_text)

    # Create TF-IDF features for test data
    X_test = vectorizer.transform(test_features_df['processed_narrative'])

    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(models, X_test)

    # Save submission
    submission_format_df = pd.read_csv(submission_format_file, index_col='uid')
    save_submission(predictions, test_features_df['uid'], submission_format_df.columns, output_submission_file)

if __name__ == "__main__":
    main()
