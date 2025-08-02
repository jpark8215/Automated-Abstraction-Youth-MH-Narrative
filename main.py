
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Custom F1 evaluation metric for XGBoost
def f1_eval_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = (y_pred > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred)


# Load the data
print("Loading data...")
train_features = pd.read_csv('assets/train_features.csv')
train_labels = pd.read_csv('assets/train_labels.csv')
test_features = pd.read_csv('data/test_features.csv')

# Create combined narratives with preprocessing
print("Creating combined narratives...")


def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Keep periods for sentence boundary detection
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text


train_features['combined_narrative'] = (
        train_features['NarrativeLE'].apply(preprocess_text) +
        ' ' +
        train_features['NarrativeCME'].apply(preprocess_text)
)
test_features['combined_narrative'] = (
        test_features['NarrativeLE'].apply(preprocess_text) +
        ' ' +
        test_features['NarrativeCME'].apply(preprocess_text)
)


def create_engineered_features(df):
    """Enhanced feature engineering with sophisticated features"""
    features = []

    # Basic text statistics
    features.append(df['combined_narrative'].str.len().values.reshape(-1, 1))
    features.append(df['combined_narrative'].str.split().str.len().values.reshape(-1, 1))
    features.append(df['combined_narrative'].str.count(r'[A-Z]').values.reshape(-1, 1))
    features.append(df['combined_narrative'].str.count(r'[!?.]').values.reshape(-1, 1))
    features.append(df['combined_narrative'].str.count(r'\d').values.reshape(-1, 1))

    # Enhanced keyword lists
    disclosure_keywords = [
        'told', 'said', 'mentioned', 'expressed', 'talked', 'discussed', 'confided',
        'shared', 'revealed', 'admitted', 'disclosed', 'informed', 'stated',
        'indicated', 'reported', 'communicate', 'conversation', 'speak'
    ]

    mental_health_keywords = [
        'depressed', 'anxious', 'bipolar', 'schizophrenia', 'ptsd', 'trauma',
        'disorder', 'medication', 'therapy', 'counseling', 'psychiatric',
        'psychologist', 'psychiatrist', 'mental health', 'treatment', 'diagnosed',
        'depression', 'anxiety', 'mental', 'emotional', 'psychological'
    ]

    substance_keywords = [
        'alcohol', 'drug', 'substance', 'addiction', 'abuse', 'dependence',
        'overdose', 'intoxicated', 'drunk', 'high', 'withdrawal', 'sober',
        'recovery', 'rehab', 'clean'
    ]

    relationship_keywords = [
        'marriage', 'divorce', 'separation', 'relationship', 'partner', 'spouse',
        'girlfriend', 'boyfriend', 'wife', 'husband', 'ex', 'romantic', 'intimate',
        'domestic', 'family', 'relative'
    ]

    risk_keywords = [
        'suicide', 'self-harm', 'hurt', 'pain', 'die', 'death', 'end', 'goodbye',
        'final', 'never', 'always', 'forever', 'last', 'warning', 'help', 'crisis',
        'emergency', 'attempt', 'plan', 'ideation'
    ]

    # Add keyword features with context
    for keywords in [disclosure_keywords, mental_health_keywords, substance_keywords,
                     relationship_keywords, risk_keywords]:
        features.append(df['combined_narrative'].apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in keywords)
        ).values.reshape(-1, 1))

        # Add ratio features
        features.append((df['combined_narrative'].apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in keywords)
        ) / df['combined_narrative'].str.split().str.len()).fillna(0).values.reshape(-1, 1))

    # Sentence-level features
    features.append(df['combined_narrative'].str.count(r'[.!?]+').values.reshape(-1, 1))

    # Create feature matrix
    feature_matrix = np.hstack(features)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    return csr_matrix(feature_matrix)


# Enhanced TF-IDF vectorizer
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=25000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w+|\?|\!|\.|,|;|:',
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


def balanced_sampling(X, y, n_samples=None):
    """Perform balanced sampling with dynamic sample size"""
    if n_samples is None:
        minority_count = min(sum(y == 0), sum(y == 1))
        n_samples = max(2000, minority_count * 2)

    # Convert sparse matrix to dense for sampling
    X_dense = X.toarray()
    combined_data = pd.DataFrame(X_dense)
    combined_data['target'] = y

    # Separate classes
    df_majority = combined_data[combined_data['target'] == 0]
    df_minority = combined_data[combined_data['target'] == 1]

    # Sampling
    if len(df_majority) > n_samples:
        df_majority = df_majority.sample(n=n_samples, random_state=42)
    if len(df_minority) > n_samples:
        df_minority = df_minority.sample(n=n_samples, random_state=42)
    else:
        df_minority = df_minority.sample(n=n_samples, replace=True, random_state=42)

    # Combine samples
    df_balanced = pd.concat([df_majority, df_minority])
    y_balanced = df_balanced['target']
    X_balanced = df_balanced.drop('target', axis=1)

    return csr_matrix(X_balanced), y_balanced


# Define columns
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

# Train binary models with optimizations
print("Training binary classification models...")
binary_predictions = {}

for col in binary_columns:
    print(f"\nTraining model for {col}")
    y = train_labels[col]

    # Calculate optimal sample size
    minority_count = min(sum(y == 0), sum(y == 1))
    sample_size = max(2000, minority_count * 2)

    # Balanced sampling
    X_balanced, y_balanced = balanced_sampling(X_train, y, sample_size)

    # Model parameters based on column type
    if col.startswith('Disclosed'):
        params = {
            'objective': 'binary:logistic',
            'max_depth': 8,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'learning_rate': 0.003,
            'n_estimators': 500,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'tree_method': 'hist',
            'eval_metric': 'logloss'
        }
    else:
        params = {
            'objective': 'binary:logistic',
            'max_depth': 7,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.005,
            'n_estimators': 400,
            'gamma': 0.2,
            'reg_alpha': 0.2,
            'reg_lambda': 1.2,
            'tree_method': 'hist',
            'eval_metric': 'logloss'
        }

    # Train-validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_balanced, y_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_balanced
    )

    # Initialize and train model
    model = XGBClassifier(**params)
    model.fit(
        X_train_split,
        y_train_split,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Make predictions and evaluate F1 score on validation set
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    print(f"Validation F1 score: {val_f1:.3f}")

    # Make predictions on test set
    binary_predictions[col] = model.predict(X_test)
    print(f"Predictions distribution: {np.unique(binary_predictions[col], return_counts=True)}")


def train_categorical_model(X, y, model_name):
    print(f"\nTraining {model_name}...")

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = dict(zip(np.unique(y), class_weights))

    params = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y)),
        'max_depth': 8,
        'min_child_weight': 2,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'learning_rate': 0.005,
        'n_estimators': 500,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'tree_method': 'hist',
        'eval_metric': 'merror'
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**params)

    scores = []
    predictions = np.zeros(X_test.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X.toarray(), y)):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        sample_weights = np.array([weight_dict[cls] for cls in y_train_fold])

        model.fit(
            X_train_fold,
            y_train_fold,
            sample_weight=sample_weights,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        # Calculate micro-F1 score on validation set
        val_pred = model.predict(X_val_fold)
        f1_micro = f1_score(y_val_fold, val_pred, average='micro')
        scores.append(f1_micro)

        fold_predictions = model.predict(X_test)
        predictions += fold_predictions

    # Average predictions from all folds
    predictions = np.round(predictions / 5).astype(int)
    print(f"{model_name} average CV micro-F1: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")

    return predictions


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

# Create submission
print("\nCreating submission file...")
submission_format = pd.read_csv('data/submission_format.csv')
submission = pd.DataFrame({'uid': test_features['uid']})

# Add binary predictions
for col in binary_columns:
    submission[col] = binary_predictions[col]

# Add categorical predictions
submission['InjuryLocationType'] = injury_location_pred
submission['WeaponType1'] = weapon_type_pred

# Ensure format matches
submission = submission_format[['uid']].merge(submission, on='uid', how='left')
submission = submission.fillna(0)

# Convert to correct data types
for col in submission.columns:
    if col != 'uid':
        submission[col] = submission[col].astype(int)

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved successfully!")


