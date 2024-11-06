import re

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from tqdm import tqdm

# Mapping dictionaries remain unchanged
injury_location_mapping = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
}

weapon_type_mapping = {i: i for i in range(1, 13)}

# weapon_type_mapping = {
#     1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
#     7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12
# }


def preprocess_text(text):
    """Preprocess text by converting to lowercase, removing special characters."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\w\s]', '', text)
    return text


def get_sentiment_score(text):
    """Calculate the sentiment polarity of a text."""
    return TextBlob(text).sentiment.polarity


def load_data(features_file, labels_file=None):
    """Load feature and label data from CSV files."""
    features_df = pd.read_csv(features_file)
    if labels_file is not None:
        labels_df = pd.read_csv(labels_file)
        return features_df, labels_df
    else:
        return features_df, None


def map_categorical_columns(df):
    """Map integer values to categorical labels for InjuryLocationType and WeaponType1 if they exist in the DataFrame."""
    if 'InjuryLocationType' in df.columns:
        df['InjuryLocationType'] = df['InjuryLocationType'].map(injury_location_mapping)
    if 'WeaponType1' in df.columns:
        df['WeaponType1'] = df['WeaponType1'].map(weapon_type_mapping)
    return df


def prepare_data(features_df, labels_df):
    """Prepare data by combining narratives, preprocessing, and mapping categorical columns."""
    features_df['processed_narrative'] = (features_df['NarrativeLE'].fillna('') + ' ' +
                                          features_df['NarrativeCME'].fillna(''))
    features_df['processed_narrative'] = features_df['processed_narrative'].apply(preprocess_text)
    features_df['sentiment'] = features_df['processed_narrative'].apply(get_sentiment_score)

    labels_df = map_categorical_columns(labels_df)
    merged_df = pd.merge(features_df[['uid', 'processed_narrative', 'sentiment']], labels_df,
                         on='uid', how='inner')
    return merged_df


def create_tfidf_features(texts, max_features=25000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    return vectorizer.fit_transform(texts), vectorizer


def analyze_class_distribution(y, column_name):
    """Analyze class distribution for a given column."""
    class_counts = y[column_name].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    print(f"\nClass Distribution for {column_name}:")
    print(class_counts)
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

    return imbalance_ratio, class_counts


def handle_imbalance(X, y, class_counts, sampling_ratio=0.8):
    """Handle class imbalance using a combination of undersampling and synthetic samples."""
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[1]

    # Get indices for each class
    maj_indices = y[y == majority_class].index
    min_indices = y[y == minority_class].index

    # Undersample majority class
    target_maj_samples = int(len(min_indices) / sampling_ratio)
    maj_indices = np.random.choice(maj_indices, target_maj_samples, replace=False)

    # Combine indices
    balanced_indices = np.concatenate([maj_indices, min_indices])

    return X[balanced_indices], y[balanced_indices]


def calculate_optimal_threshold(y_true, y_pred_proba):
    """Calculate optimal threshold to minimize false positives."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5


def evaluate_predictions(y_true, y_pred, y_pred_proba, column_name):
    """Evaluate model predictions with focus on false positives."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, average='binary')
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\nMetrics for {column_name}:")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")

def train_and_evaluate_model(X, y):
    """Enhanced model training with focus on reducing false positives."""
    models = []
    scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for col in tqdm(y.columns, desc="Training models"):
        print(f"\n*****Processing {col}*****")

        # Analyze class distribution
        imbalance_ratio, class_counts = analyze_class_distribution(y, col)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y[col])

        if len(np.unique(y_encoded)) < 2:
            print(f"Skipping {col} due to insufficient class samples.")
            continue

        # Calculate class weights
        class_counts_encoded = np.bincount(y_encoded)
        class_weight = {
            i: len(y_encoded) / (len(np.unique(y_encoded)) * count)
            for i, count in enumerate(class_counts_encoded)
        }

        # Define base models
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=(len(y_encoded) / sum(y_encoded)) if len(np.unique(y_encoded)) == 2 else 1,
            # scale_pos_weight=class_counts_encoded[0] / class_counts_encoded[1] if len(class_counts_encoded) == 2 else 1,
            random_state=42
        )

        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.7,
            random_state=42
        )

        # Perform cross-validation
        cv_scores = []
        thresholds = []

        for train_idx, val_idx in skf.split(X.toarray(), y_encoded):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            # Handle imbalance if binary classification
            if len(np.unique(y_encoded)) == 2 and imbalance_ratio > 3:
                X_train_dense = X_train.toarray()
                X_train_balanced, y_train_balanced = handle_imbalance(
                    X_train_dense,
                    pd.Series(y_train),
                    pd.Series(y_train).value_counts()
                )
                X_train = csr_matrix(X_train_balanced)
                y_train = y_train_balanced.values

            # Train models
            models_list = [rf_model, xgb_model, gb_model]
            predictions_proba = []

            for model in models_list:
                model.fit(X_train, y_train)
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_val)[:, 1]
                    predictions_proba.append(pred_proba)

            # Ensemble predictions
            if predictions_proba:
                y_pred_proba = np.mean(predictions_proba, axis=0)
                threshold = calculate_optimal_threshold(y_val, y_pred_proba)
                thresholds.append(threshold)
                y_pred = (y_pred_proba >= threshold).astype(int)
            else:
                y_pred = rf_model.predict(X_val)

            cv_scores.append(f1_score(y_val, y_pred, average='weighted'))

            if len(np.unique(y_encoded)) == 2:
                evaluate_predictions(y_val, y_pred, y_pred_proba, col)

        # Train final models on full dataset
        final_models = []
        for model in models_list:
            model.fit(X.toarray(), y_encoded)
            final_models.append(model)

        final_threshold = np.mean(thresholds) if thresholds else None
        models.append((final_models, label_encoder, final_threshold))
        scores.append(np.mean(cv_scores))

    return models, np.mean(scores)


def make_predictions(models, X):
    """Make predictions using the trained models."""
    predictions = []
    for model_list, label_encoder, threshold in models:
        predictions_proba = []

        for model in model_list:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
                predictions_proba.append(pred_proba)

        if predictions_proba and threshold is not None:
            # Ensemble predictions for binary classification
            y_pred_proba = np.mean(predictions_proba, axis=0)
            pred = (y_pred_proba >= threshold).astype(int)
        else:
            # Use random forest for multi-class or when probability is not available
            pred = model_list[0].predict(X)

        pred = label_encoder.inverse_transform(pred)
        predictions.append(pred)

    return np.column_stack(predictions)


def save_submission(predictions, uids, columns, output_file):
    """Save predictions to a submission file with UID as index."""
    submission_df = pd.DataFrame(predictions, columns=columns)
    submission_df['uid'] = uids
    submission_df.set_index('uid', inplace=True)
    submission_df.to_csv(output_file)
    print(f"Submission saved to {output_file}")


def main():
    train_features_file = 'assets/train_features.csv'
    train_labels_file = 'assets/train_labels.csv'
    test_features_file = 'data/test_features.csv'
    submission_format_file = 'data/submission_format.csv'
    output_submission_file = 'submission.csv'

    print("Loading and preparing training data...")
    train_features_df, train_labels_df = load_data(train_features_file, train_labels_file)
    train_df = prepare_data(train_features_df, train_labels_df)

    print("Creating TF-IDF features...")
    X_train, vectorizer = create_tfidf_features(train_df['processed_narrative'])
    y_train = train_df.drop(['uid', 'processed_narrative', 'sentiment'], axis=1)

    print("Training and evaluating models...")
    models, avg_score = train_and_evaluate_model(X_train, y_train)

    print("Loading and preparing test data...")
    test_features_df, _ = load_data(test_features_file, None)
    test_features_df['processed_narrative'] = (test_features_df['NarrativeLE'].fillna('') + ' ' +
                                               test_features_df['NarrativeCME'].fillna(''))
    test_features_df['processed_narrative'] = test_features_df['processed_narrative'].apply(preprocess_text)
    test_features_df['sentiment'] = test_features_df['processed_narrative'].apply(get_sentiment_score)

    X_test = vectorizer.transform(test_features_df['processed_narrative'])

    print("Making predictions...")
    predictions = make_predictions(models, X_test)

    submission_format_df = pd.read_csv(submission_format_file, index_col='uid')
    save_submission(predictions, test_features_df['uid'], submission_format_df.columns, output_submission_file)


if __name__ == "__main__":
    main()

