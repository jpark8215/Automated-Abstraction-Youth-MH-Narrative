import logging
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define categorical columns and their possible values
CATEGORICAL_COLS = {
    'InjuryLocationType': range(1, 7),  # 1-6
    'WeaponType1': range(1, 13)  # 1-12
}


def load_data(features_path, labels_path):
    logging.info(f"Loading data from {features_path} and {labels_path}")
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    return pd.merge(features, labels, on='uid', how='inner')


def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    # Keep some punctuation that might be meaningful
    text = re.sub(r'[^a-z0-9\s\.\,\?\!]', '', text)
    text = re.sub(r'\s\^xxxx+', ' ', text).strip()
    return text


def prepare_data(df):
    """Prepare data by combining narratives and preprocessing."""
    df = df.copy()

    # Combine narratives with special separator
    df['processed_narrative'] = (df['NarrativeLE'].fillna('') +
                                 ' [SEP] ' +
                                 df['NarrativeCME'].fillna(''))

    df['processed_narrative'] = df['processed_narrative'].apply(preprocess_text)

    # Remove rows where narrative is empty
    df = df[df['processed_narrative'].str.strip() != '']

    return df


def engineer_features(X_train, X_val=None, max_features=25000, n_components=250):
    logging.info(f"Starting feature engineering with TF-IDF and TruncatedSVD.")

    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True  # Often helps in reducing the impact of very frequent terms
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    svd = TruncatedSVD(n_components=min(n_components, X_train_tfidf.shape[1] - 1),
                       random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)

    explained_var = svd.explained_variance_ratio_.sum()
    logging.info(f"Explained variance ratio with SVD: {explained_var:.4f}")

    if X_val is not None:
        X_val_tfidf = tfidf.transform(X_val)
        X_val_svd = svd.transform(X_val_tfidf)
        return X_train_svd, X_val_svd, tfidf, svd

    return X_train_svd, tfidf, svd


class MixedClassifier:
    """Custom classifier that handles both binary and categorical predictions"""

    def __init__(self, categorical_cols, custom_thresholds=None):
        self.categorical_cols = categorical_cols
        self.binary_cols = None
        self.binary_model = None
        self.categorical_models = {}
        self.custom_thresholds = custom_thresholds if custom_thresholds else {}

    def fit(self, X, y):
        # Split columns into binary and categorical
        self.binary_cols = [col for col in y.columns if col not in self.categorical_cols]

        # Train binary classifier using MultiOutputClassifier
        if self.binary_cols:
            base_rf = RandomForestClassifier(
                n_estimators=1000,
                max_depth=50,
                min_samples_split=9,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.binary_model = MultiOutputClassifier(base_rf)
            self.binary_model.fit(X, y[self.binary_cols])

        # Train categorical classifiers
        for col in self.categorical_cols:
            self.categorical_models[col] = RandomForestClassifier(
                n_estimators=500,
                max_depth=30,
                min_samples_split=27,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            # Ensure values are integers and handle missing values
            y_cat = y[col].fillna(-1).astype(int)
            self.categorical_models[col].fit(X, y_cat)

    def predict(self, X):
        predictions = {}

        # Binary predictions
        if self.binary_cols:
            # Get probabilities for all binary columns
            binary_probs = self.binary_model.predict_proba(X)

            # Convert probabilities to predictions using custom thresholds
            for i, col in enumerate(self.binary_cols):
                # Get probability of class 1 for current column
                col_probs = binary_probs[i][:, 1]

                threshold = self.custom_thresholds.get(col, 0.4)
                predictions[col] = (col_probs > threshold).astype(int)

        # Categorical predictions
        for col in self.categorical_cols:
            preds = self.categorical_models[col].predict(X)
            # Ensure predictions are valid categories
            valid_categories = list(self.categorical_cols[col])
            predictions[col] = np.clip(preds, min(valid_categories), max(valid_categories))

        return pd.DataFrame(predictions)


def train_model(X_train, y_train, X_val=None, y_val=None):
    logging.info("Training Mixed Classifier for binary and categorical predictions")

    # Define custom thresholds for specific columns
    custom_thresholds = {
        'DepressedMood': 0.35,
        'MentalIllnessTreatmentCurrnt': 0.37,
        'SuicideAttemptHistory': 0.3,
        'SubstanceAbuseProblem': 0.32,
        'DiagnosisAnxiety': 0.25,
        'DiagnosisDepressionDysthymia': 0.38,
        'DiagnosisBipolar': 0.19,
        'DiagnosisAdhd': 0.2,
        'IntimatePartnerProblem': 0.4,
        'FamilyRelationship': 0.3,
        'Argument': 0.34,
        'SchoolProblem': 0.23,
        'RecentCriminalLegalProblem': 0.19,
        'SuicideNote': 0.39,
        'SuicideIntentDisclosed': 0.32,
        'DisclosedToIntimatePartner': 0.23,
        'DisclosedToOtherFamilyMember': 0.31,
        'DisclosedToFriend': 0.25
    }

    model = MixedClassifier(CATEGORICAL_COLS, custom_thresholds)
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)

        # Compute metrics for each column
        for col in y_train.columns:
            if col in CATEGORICAL_COLS:
                f1 = f1_score(y_val[col], y_val_pred[col], average='weighted')

            else:
                f1 = f1_score(y_val[col], y_val_pred[col], average='binary')
            logging.info(f"F1 Score for {col}: {f1:.4f}")

    return model


def predict_and_create_submission(model, X_test, tfidf, svd, test_features, submission_format_path, output_path):
    logging.info("Transforming test data and making predictions.")

    # Prepare test data
    test_features['processed_narrative'] = (
            test_features['NarrativeLE'].fillna('') +
            ' [SEP] ' +
            test_features['NarrativeCME'].fillna('')
    )
    test_features['processed_narrative'] = test_features['processed_narrative'].apply(preprocess_text)

    # Transform test data
    X_test_tfidf = tfidf.transform(test_features['processed_narrative'])
    X_test_svd = svd.transform(X_test_tfidf)

    # Make predictions
    predictions = model.predict(X_test_svd)

    # Add uid column
    predictions.insert(0, 'uid', test_features['uid'])

    # Load submission format to ensure correct column order
    submission_format = pd.read_csv(submission_format_path)
    predictions = predictions[submission_format.columns]

    # Save submission
    predictions.to_csv(output_path, index=False)
    logging.info(f"Submission saved to {output_path}")


def main():
    logging.info("Starting the pipeline.")

    # Load and prepare data
    df = load_data('assets/train_features.csv', 'assets/train_labels.csv')
    prepared_data = prepare_data(df)

    # Split features and labels
    X = prepared_data['processed_narrative']
    y = prepared_data.drop(columns=['uid', 'processed_narrative', 'NarrativeLE', 'NarrativeCME'])

    # Convert labels to numeric
    y = y.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['WeaponType1']
    )

    # Engineer features
    X_train_features, X_val_features, tfidf, svd = engineer_features(X_train, X_val)

    # Train and evaluate model
    model = train_model(X_train_features, y_train, X_val_features, y_val)

    # Load and prepare test data
    test_features = pd.read_csv('data/test_features.csv')

    # Create submission
    predict_and_create_submission(
        model, None, tfidf, svd, test_features,
        'data/submission_format.csv', 'submission.csv'
    )

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
