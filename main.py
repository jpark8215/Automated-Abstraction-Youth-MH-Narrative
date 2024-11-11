import logging
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm

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
    def __init__(self, categorical_cols, metric='f1', n_splits=5, seed=42, num_targets=None):
        self.categorical_cols = categorical_cols
        self.metric = metric
        self.n_splits = n_splits
        self.seed = seed
        self.models = {}
        self.num_targets = num_targets

    def get_metric(self, y_true, y_pred):
        """Calculate metric for each target column individually and average them."""
        # Ensure y_true and y_pred are DataFrames for consistent access
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_frame()
        if isinstance(y_pred, np.ndarray) and y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        elif isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_frame()

        scores = []
        for i in range(y_true.shape[1]):  # Loop over each target column
            col_y_true = y_true.iloc[:, i]
            col_y_pred = y_pred[:, i]

            # Compute the metric for each column based on the specified metric
            if self.metric == 'f1':
                score = f1_score(col_y_true, col_y_pred, average='weighted')
            elif self.metric == 'accuracy':
                score = accuracy_score(col_y_true, col_y_pred)
            elif self.metric == 'roc_auc':
                score = roc_auc_score(col_y_true, col_y_pred, multi_class='ovr')
            else:
                raise ValueError(f"Unsupported metric '{self.metric}'")

            scores.append(score)

        # Return the average score across all target columns
        return np.mean(scores)

    def train(self, X, y):
        if self.num_targets is None:
            self.num_targets = y.shape[1]

        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof_predictions = np.zeros(y.shape)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X), desc="Training Folds")):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # MultiOutputClassifier for handling each binary/categorical target
            model = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=self.seed, class_weight="balanced"))
            model.fit(X_train, y_train)
            self.models[fold] = model

            # Predict on validation set
            y_val_pred = model.predict(X_val)
            oof_predictions[val_idx] = y_val_pred

            # Average score across all target columns
            fold_scores = [self.get_metric(y_val.iloc[:, i], y_val_pred[:, i]) for i in range(y.shape[1])]
            avg_fold_score = np.mean(fold_scores)
            scores.append(avg_fold_score)
            logging.info(f"Fold {fold + 1} - Average {self.metric.upper()}: {avg_fold_score:.4f}")

        avg_score = np.mean(scores)
        logging.info(f"Overall CV {self.metric.upper()}: {avg_score:.4f}")
        return oof_predictions, avg_score

    def predict(self, X_test):
        if self.num_targets is None:
            raise ValueError("num_targets must be defined. Train the model first or specify it during initialization.")

        test_preds = np.zeros((X_test.shape[0], self.num_targets, self.n_splits))

        for fold, model in self.models.items():
            test_preds[:, :, fold] = model.predict(X_test)

        return test_preds.mean(axis=2).round().astype(int)


def predict_and_create_submission(model, X_test, tfidf, svd, test_features, submission_format_path, output_path, target_columns):
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

    # Convert predictions to a DataFrame and add 'uid' column
    predictions = pd.DataFrame(predictions, columns=target_columns)
    predictions.insert(0, 'uid', test_features['uid'])

    # Load submission format to ensure correct column order
    submission_format = pd.read_csv(submission_format_path)
    predictions = predictions[submission_format.columns]

    # Save submission
    predictions.to_csv(output_path, index=False)
    logging.info(f"Submission saved to {output_path}")




def main():
    logging.info("Starting the pipeline.")

    # Load data
    df = load_data('assets/train_features.csv', 'assets/train_labels.csv')
    prepared_data = prepare_data(df)

    # Split features and labels
    X = prepared_data['processed_narrative']
    y = prepared_data.drop(columns=['uid', 'processed_narrative', 'NarrativeLE', 'NarrativeCME'])
    y = y.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Engineer features
    X_train_features, X_val_features, tfidf, svd = engineer_features(X_train, X_val)

    # Train model
    mixed_model = MixedClassifier(categorical_cols=CATEGORICAL_COLS, metric='f1', n_splits=5, seed=42)
    oof_preds, avg_cv_score = mixed_model.train(X_train_features, y_train)
    logging.info(f"Average cross-validation {mixed_model.metric.upper()}: {avg_cv_score:.4f}")

    # Validate model on held-out data
    y_val_pred = mixed_model.predict(X_val_features)
    val_scores = [mixed_model.get_metric(y_val.iloc[:, i], y_val_pred[:, i]) for i in range(y_val.shape[1])]
    avg_val_score = np.mean(val_scores)
    logging.info(f"Validation {mixed_model.metric.upper()}: {avg_val_score:.4f}")

    # Load and process test data
    test_features = pd.read_csv('data/test_features.csv')
    test_features['processed_narrative'] = (
            test_features['NarrativeLE'].fillna('') + ' [SEP] ' + test_features['NarrativeCME'].fillna('')
    )
    test_features['processed_narrative'] = test_features['processed_narrative'].apply(preprocess_text)
    X_test_tfidf = tfidf.transform(test_features['processed_narrative'])
    X_test_svd = svd.transform(X_test_tfidf)

    # Generate and save test predictions
    # test_predictions = mixed_model.predict(X_test_svd)
    predict_and_create_submission(
        model=mixed_model,
        X_test=X_test_svd,
        tfidf=tfidf,
        svd=svd,
        test_features=test_features,
        submission_format_path='data/submission_format.csv',
        output_path='submission.csv',
        target_columns=y_train.columns  # Pass the correct target columns
    )

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
