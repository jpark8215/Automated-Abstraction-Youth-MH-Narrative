import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Load the data
train_features = pd.read_csv('assets/train_features.csv')
train_labels = pd.read_csv('assets/train_labels.csv')
test_features = pd.read_csv('data/test_features.csv')

# Create combined narratives first
print("Creating combined narratives...")
train_features['combined_narrative'] = train_features['NarrativeLE'] + ' ' + train_features['NarrativeCME']
test_features['combined_narrative'] = test_features['NarrativeLE'] + ' ' + test_features['NarrativeCME']


def create_engineered_features(df):
    """Create engineered features from the narrative text and other columns"""
    features = []

    # Text length features
    features.append(df['combined_narrative'].str.len().values.reshape(-1, 1))
    features.append(df['combined_narrative'].str.split().str.len().values.reshape(-1, 1))

    # Specific content indicators
    disclosure_keywords = ['told', 'said', 'mentioned', 'expressed', 'talked', 'discussed', 'confided', 'shared',
                           'revealed']
    features.append(df['combined_narrative'].apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in disclosure_keywords)
    ).values.reshape(-1, 1))

    family_keywords = ['family', 'mother', 'father', 'sister', 'brother', 'daughter', 'son',
                       'wife', 'husband', 'spouse', 'parent', 'child', 'sibling']
    features.append(df['combined_narrative'].apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in family_keywords)
    ).values.reshape(-1, 1))

    friend_keywords = ['friend', 'colleague', 'coworker', 'neighbor', 'peer', 'acquaintance',
                       'buddy', 'companion', 'roommate', 'classmate']
    features.append(df['combined_narrative'].apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in friend_keywords)
    ).values.reshape(-1, 1))

    # Emotional content indicators
    emotion_keywords = ['depressed', 'sad', 'angry', 'upset', 'stressed', 'anxious', 'worried',
                        'frustrated', 'hopeless', 'helpless', 'lonely', 'isolated']
    features.append(df['combined_narrative'].apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in emotion_keywords)
    ).values.reshape(-1, 1))

    # Communication method indicators
    communication_keywords = ['text', 'call', 'phone', 'message', 'email', 'letter', 'note',
                              'wrote', 'called', 'texted', 'messaged', 'contacted']
    features.append(df['combined_narrative'].apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in communication_keywords)
    ).values.reshape(-1, 1))

    # Time indicators
    time_keywords = ['today', 'yesterday', 'week', 'month', 'recent', 'lately', 'previously']
    features.append(df['combined_narrative'].apply(
        lambda x: sum(1 for word in str(x).lower().split() if word in time_keywords)
    ).values.reshape(-1, 1))

    # Stack all features horizontally
    feature_matrix = np.hstack(features)

    # Scale the features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    return csr_matrix(feature_matrix)


# Create TF-IDF vectors
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Process training data
print("Processing training data...")
X_train_tfidf = vectorizer.fit_transform(train_features['combined_narrative'])
train_engineered = create_engineered_features(train_features)
X_train = hstack([X_train_tfidf, train_engineered])

# Process test data
print("Processing test data...")
X_test_tfidf = vectorizer.transform(test_features['combined_narrative'])
test_engineered = create_engineered_features(test_features)
X_test = hstack([X_test_tfidf, test_engineered])

# Define the columns
binary_columns = [
    'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt',
    'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem',
    'MentalHealthProblem', 'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia',
    'DiagnosisBipolar', 'DiagnosisAdhd', 'IntimatePartnerProblem',
    'FamilyRelationship', 'Argument', 'SchoolProblem', 'RecentCriminalLegalProblem',
    'SuicideNote', 'SuicideIntentDisclosed', 'DisclosedToIntimatePartner',
    'DisclosedToOtherFamilyMember', 'DisclosedToFriend'
]

categorical_columns = ['InjuryLocationType', 'WeaponType1']


# Enhanced binary classification with undersampling
print("Training binary classification models with undersampling...")
binary_predictions = {}

for col in binary_columns:
    print(f"\nTraining model for {col}")
    y = train_labels[col]

    # Initialize the undersampler
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y)

    # Display class distribution after undersampling
    print(f"Class distribution for {col} after undersampling: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

    # Calculate class weights for the resampled data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
    class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))

    # Define model parameters with class weights
    if col.startswith('Disclosed'):
        # Custom parameters for disclosure-related columns
        params = {
            'objective': 'binary:logistic',
            'max_depth': 8,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'learning_rate': 0.005,
            'n_estimators': 300,
            'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],
            'tree_method': 'hist'
        }
    else:
        # Default parameters for other columns
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],
            'tree_method': 'hist'
        }

    # Initialize and train the model
    binary_model = XGBClassifier(**params)

    # Cross-validation with the undersampled data
    cv_scores = cross_val_score(
        binary_model, X_resampled, y_resampled,
        cv=StratifiedKFold(n_splits=5),
        scoring='f1'
    )
    print(
        f"Cross-validation F1 scores for {col} after undersampling: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Fit the model on the resampled data
    binary_model.fit(X_resampled, y_resampled)

    # Predict on the test data
    binary_predictions[col] = binary_model.predict(X_test)


# Train categorical models
def train_categorical_model(X, y, model_name):
    print(f"\nTraining {model_name}...")

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = dict(zip(np.unique(y), class_weights))
    sample_weights = np.array([weight_dict[cls] for cls in y])

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        max_depth=7,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.01,
        n_estimators=300,
        tree_method='hist'
    )

    cv_scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits=5),
        scoring='f1_weighted'
    )
    print(f"Cross-validation F1 scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    model.fit(X, y, sample_weight=sample_weights)
    return model.predict(X_test)


# Train categorical models
print("\nTraining categorical models...")
injury_location_pred = train_categorical_model(
    X_train,
    train_labels['InjuryLocationType'] - 1,
    'Injury Location'
) + 1

weapon_type_pred = train_categorical_model(
    X_train,
    train_labels['WeaponType1'] - 1,
    'Weapon Type'
) + 1

# Ensure uid alignment between test features and submission format
submission_format = pd.read_csv('data/submission_format.csv')
submission = pd.DataFrame({'uid': test_features['uid']})

# Add predictions for binary columns to the submission DataFrame
for col in binary_columns:
    if col in binary_predictions:
        submission[col] = binary_predictions[col]
    else:
        submission[col] = 0  # Default to 0 if prediction is missing

# Add categorical predictions
submission['InjuryLocationType'] = injury_location_pred
submission['WeaponType1'] = weapon_type_pred

# Ensure all columns in the submission format are present in `submission`
submission = submission_format[['uid']].merge(submission, on='uid', how='left')

# Verify column and row alignment by ensuring each submission column exists and is in the right order
for col in submission_format.columns:
    if col not in submission.columns:
        submission[col] = 0  # Add missing columns and set default values
    else:
        submission[col].fillna(0, inplace=True)  # Fill any NaN values with 0

# Ensure correct data types
for col in submission.columns:
    if col != 'uid':
        submission[col] = submission[col].astype(int)

# Save the final submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully and verified for correct structure!")
