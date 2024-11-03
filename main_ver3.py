import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from tqdm import tqdm
import re
import xgboost as xgb

# Mapping dictionaries for categorical columns
injury_location_mapping = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6
}

weapon_type_mapping = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12
}

keyword_mappings = {
    "injury_location": {
        "house|apartment": 1,
        "vehicle|car|motorcycle": 2,
        "river|field|beach|wood": 3,
        "park|playground|public": 4,
        "street|road|sidewalk|alley": 5,
        "other": 6
    },
    "weapon_type": {
        "hammer|wrenches|pipe|stick|bat|scissors": 1,
        "drowning": 2,
        "fall": 3,
        "fire|burns": 4,
        "Firearm|gun|shot": 5,
        "hanging|strangul|suffocat": 6,
        "motor|vehicle|buses|motorcycles": 7,
        "trains|planes|boats": 8,
        "poison": 9,
        "sharp|knife": 10,
        "other": 11,
        "Unknown": 12
    }
}


def preprocess_text(text):
    """Preprocess text by converting to lowercase, removing special characters."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
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
    merged_df = pd.merge(features_df[['uid', 'processed_narrative', 'sentiment']], labels_df, on='uid', how='inner')
    return merged_df


def create_tfidf_features(texts, max_features=10000):
    """Create TF-IDF features from preprocessed texts."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    return vectorizer.fit_transform(texts), vectorizer


def train_and_evaluate_model(X, y):
    """Train an ensemble model for each label and evaluate using cross-validation."""
    models = []
    scores = []

    for col in tqdm(y.columns, desc="Training models"):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y[col])

        if len(np.unique(y_encoded)) < 2:
            print(f"Skipping {col} due to insufficient class samples.")
            continue

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
        ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')

        ensemble_score = cross_val_score(ensemble_model, X, y_encoded, cv=3, scoring='f1_micro').mean()
        ensemble_model.fit(X, y_encoded)
        models.append((ensemble_model, label_encoder))
        scores.append(ensemble_score)
        print(f"Ensemble model F1 score for {col}: {ensemble_score * 100:.2f}%")

    avg_score = np.mean(scores)
    print(f"Average F1 score across all labels: {avg_score * 100:.2f}%")
    return models, avg_score


def make_predictions(models, X):
    """Make predictions using the trained models."""
    predictions = []
    for model, label_encoder in models:
        pred = model.predict(X)
        pred = label_encoder.inverse_transform(pred)
        predictions.append(pred)
    return np.column_stack(predictions)


def adjust_predictions_based_on_keywords(df, predictions, keyword_mappings):
    """ Adjust predictions based on narrative content for injury locations and weapon types. """
    adjusted_predictions = predictions.copy()

    for i, narrative in enumerate(df['processed_narrative']):
        for category, keywords in keyword_mappings['injury_location'].items():
            if re.search(category, narrative):
                adjusted_predictions[i, df.columns.get_loc('InjuryLocationType')] = keywords

        for category, keywords in keyword_mappings['weapon_type'].items():
            if re.search(category, narrative):
                adjusted_predictions[i, df.columns.get_loc('WeaponType1')] = keywords

    return adjusted_predictions


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
    # Adjust predictions based on keyword mappings
    # predictions = adjust_predictions_based_on_keywords(test_features_df, predictions, keyword_mappings)

    submission_format_df = pd.read_csv(submission_format_file, index_col='uid')
    save_submission(predictions, test_features_df['uid'], submission_format_df.columns, output_submission_file)


if __name__ == "__main__":
    main()