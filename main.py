
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from tqdm import tqdm
import re
import xgboost as xgb

# Mapping dictionaries for categorical columns
injury_location_mapping = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
}

weapon_type_mapping = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12
}


# # Define your keyword mappings here
# keyword_mappings = {
#     "injury_location": {
#         r"house|apartment|condo|flat|home": 1,
#         r"car|motorcycle|van|truck|vehicle": 2,
#         r"river|field|beach|wood|forest|park|countryside": 3,
#         r"park|playground|public space|community": 4,
#         r"street|road|sidewalk|alley|driveway|highway": 5,
#         r"unknown|other|not specified": 6
#     },
#     "weapon_type": {
#         r"hammer|wrench|pipe|stick|bat|scissors|tool": 1,
#         r"drowning|water|submersion": 2,
#         r"fall|trip|slip|accidental fall": 3,
#         r"fire|burns|explosion|flame": 4,
#         r"firearm|gun|shot|pistol|revolver|rifle": 5,
#         r"hanging|strangul|suffocat|asphyxiat": 6,
#         r"motor|vehicle|bus|motorcycle|car": 7,
#         r"train|plane|boat|aircraft": 8,
#         r"poison|overdose|toxic|substance": 9,
#         r"sharp|knife|blade|sword|scissors|cutting tool": 10,
#         r"taser|elect|nail gun|stun gun": 11,
#         r"unknown|unspecified|not known": 12
#     }
# }
#
# # Define boolean keywords
# boolean_keywords = {
#     "DepressedMood": r"depress|sad|unhappy",
#     "MentalIllnessTreatmentCurrnt": r"current treatment|current therapy|ongoing treatment|current counseling|present treatment",
#     "HistoryMentalIllnessTreatmnt": r"previous treatment|past treatment|former treatment|prior counseling|previous therapy",
#     "SuicideAttemptHistory": r"attempted suicide|suicide attempt|tried to kill|self-harm",
#     "SuicideThoughtHistory": r"suicidal thought|suicidal plan|thoughts of suicide|considering suicide|suicidal ideation",
#     "SubstanceAbuseProblem": r"substance|alcohol|drug|addiction|dependency",
#     "MentalHealthProblem": r"mental health|diagnos|psychological issue|psychological disorder",
#     "DiagnosisAnxiety": r"anxiety|anxious|panic|nervous|worry",
#     "DiagnosisDepressionDysthymia": r"depress|dysthymia|major depressive|depressive disorder|chronic depression",
#     "DiagnosisBipolar": r"bipolar|manic|mood swings|manic depressive",
#     "DiagnosisAdhd": r"adhd|attention deficit|hyperactivity|attention deficit hyperactivity disorder",
#     "IntimatePartnerProblem": r"partner problem|relationship issue|domestic dispute|intimate partner violence",
#     "FamilyRelationship": r"family conflict|domestic issue|family issue|family problems|home conflict",
#     "Argument": r"argu|disput|fight|quarrel|altercat",
#     "SchoolProblem": r"school issue|academic problem|educational issue|school conflict",
#     "RecentCriminalLegalProblem": r"criminal|legal issue|arrest|conviction|charge|legal trouble",
#     "SuicideNote": r"suicide note|final message|note left behind|last word",
#     "SuicideIntentDisclosed": r"disclose intent|tell someone about suicide|mention suicide|express suicide",
#     "DisclosedToIntimatePartner": r"tell partner|partner aware|tell husband|tell wife|share with partner",
#     "DisclosedToOtherFamilyMember": r"family member aware|tell family|inform family member",
#     "DisclosedToFriend": r"friend aware|tell friend|inform friend|share with friend"
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

    # # Initialize columns for boolean flags
    # for key in boolean_keywords.keys():
    #     features_df[key] = 0  # Default to 0

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


def train_and_evaluate_model(X, y):
    """Enhanced model training with confusion matrix analysis."""
    models = []
    scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for col in tqdm(y.columns, desc="Training models"):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y[col])

        if len(np.unique(y_encoded)) < 2:
            print(f"Skipping {col} due to insufficient class samples.")
            continue

        # Calculate class weights
        class_counts = np.bincount(y_encoded)
        class_weight = {i: len(y_encoded) / (len(np.unique(y_encoded)) * count)
                        for i, count in enumerate(class_counts)}

        # Define base models with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_counts[0] / class_counts[1],
            random_state=42
        )

        # gb_model = GradientBoostingClassifier(
        #     n_estimators=200,
        #     max_depth=8,
        #     learning_rate=0.1,
        #     subsample=0.8,
        #     random_state=42
        # )

        # Create SMOTE pipeline
        pipeline = Pipeline([
            # ('smote', SMOTE(random_state=42)),
            ('ensemble', VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    # ('gb', gb_model)
                ],
                voting='soft'
            ))
        ])

        # Perform cross-validation
        cv_scores = []
        confusion_matrices = []

        for train_idx, val_idx in skf.split(X.toarray(), y_encoded):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            cv_scores.append(f1_score(y_val, y_pred, average='weighted'))
            confusion_matrices.append(confusion_matrix(y_val, y_pred))

        # Print metrics
        avg_score = np.mean(cv_scores)
        print(f"\nMetrics for {col}:")
        print(f"Average F1 Score: {avg_score:.4f}")
        print("\nAverage Confusion Matrix:")
        avg_conf_matrix = np.mean(confusion_matrices, axis=0)
        print(avg_conf_matrix)

        # Fit final model on full dataset
        pipeline.fit(X.toarray(), y_encoded)
        models.append((pipeline, label_encoder))
        scores.append(avg_score)

    return models, np.mean(scores)


def make_predictions(models, X):
    """Make predictions using the trained models."""
    predictions = []
    for model, label_encoder in models:
        pred = model.predict(X)
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
