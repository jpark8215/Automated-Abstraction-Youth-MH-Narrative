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




# import numpy as np
# import pandas as pd
# from scipy.sparse import hstack, csr_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from xgboost import XGBClassifier
#
# # Load the data
# print("Loading data...")
# train_features = pd.read_csv('assets/train_features.csv')
# train_labels = pd.read_csv('assets/train_labels.csv')
# test_features = pd.read_csv('data/test_features.csv')
#
# # Create combined narratives with preprocessing
# print("Creating combined narratives...")
#
#
# def preprocess_text(text):
#     """Basic text preprocessing"""
#     if pd.isna(text):
#         return ""
#     text = str(text).lower()
#     # Keep periods for sentence boundary detection
#     text = text.replace('\n', ' ').replace('\r', ' ')
#     return text
#
#
# train_features['combined_narrative'] = (
#         train_features['NarrativeLE'].apply(preprocess_text) +
#         ' ' +
#         train_features['NarrativeCME'].apply(preprocess_text)
# )
# test_features['combined_narrative'] = (
#         test_features['NarrativeLE'].apply(preprocess_text) +
#         ' ' +
#         test_features['NarrativeCME'].apply(preprocess_text)
# )
#
#
# def create_engineered_features(df):
#     """Enhanced feature engineering with sophisticated features"""
#     features = []
#
#     # Basic text statistics
#     features.append(df['combined_narrative'].str.len().values.reshape(-1, 1))
#     features.append(df['combined_narrative'].str.split().str.len().values.reshape(-1, 1))
#     features.append(df['combined_narrative'].str.count(r'[A-Z]').values.reshape(-1, 1))
#     features.append(df['combined_narrative'].str.count(r'[!?.]').values.reshape(-1, 1))
#     features.append(df['combined_narrative'].str.count(r'\d').values.reshape(-1, 1))
#
#     # Enhanced keyword lists
#     disclosure_keywords = [
#         'told', 'said', 'mentioned', 'expressed', 'talked', 'discussed', 'confided',
#         'shared', 'revealed', 'admitted', 'disclosed', 'informed', 'stated',
#         'indicated', 'reported', 'communicate', 'conversation', 'speak'
#     ]
#
#     mental_health_keywords = [
#         'depressed', 'anxious', 'bipolar', 'schizophrenia', 'ptsd', 'trauma',
#         'disorder', 'medication', 'therapy', 'counseling', 'psychiatric',
#         'psychologist', 'psychiatrist', 'mental health', 'treatment', 'diagnosed',
#         'depression', 'anxiety', 'mental', 'emotional', 'psychological'
#     ]
#
#     substance_keywords = [
#         'alcohol', 'drug', 'substance', 'addiction', 'abuse', 'dependence',
#         'overdose', 'intoxicated', 'drunk', 'high', 'withdrawal', 'sober',
#         'recovery', 'rehab', 'clean'
#     ]
#
#     relationship_keywords = [
#         'marriage', 'divorce', 'separation', 'relationship', 'partner', 'spouse',
#         'girlfriend', 'boyfriend', 'wife', 'husband', 'ex', 'romantic', 'intimate',
#         'domestic', 'family', 'relative'
#     ]
#
#     risk_keywords = [
#         'suicide', 'self-harm', 'hurt', 'pain', 'die', 'death', 'end', 'goodbye',
#         'final', 'never', 'always', 'forever', 'last', 'warning', 'help', 'crisis',
#         'emergency', 'attempt', 'plan', 'ideation'
#     ]
#
#     # Add keyword features with context
#     for keywords in [disclosure_keywords, mental_health_keywords, substance_keywords,
#                      relationship_keywords, risk_keywords]:
#         features.append(df['combined_narrative'].apply(
#             lambda x: sum(1 for word in str(x).lower().split() if word in keywords)
#         ).values.reshape(-1, 1))
#
#         # Add ratio features
#         features.append((df['combined_narrative'].apply(
#             lambda x: sum(1 for word in str(x).lower().split() if word in keywords)
#         ) / df['combined_narrative'].str.split().str.len()).fillna(0).values.reshape(-1, 1))
#
#     # Sentence-level features
#     features.append(df['combined_narrative'].str.count(r'[.!?]+').values.reshape(-1, 1))
#
#     # Create feature matrix
#     feature_matrix = np.hstack(features)
#     scaler = StandardScaler()
#     feature_matrix = scaler.fit_transform(feature_matrix)
#
#     return csr_matrix(feature_matrix)
#
#
# # Enhanced TF-IDF vectorizer
# print("Creating TF-IDF features...")
# vectorizer = TfidfVectorizer(
#     max_features=25000,
#     stop_words='english',
#     ngram_range=(1, 4),
#     min_df=3,
#     max_df=0.9,
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='word',
#     token_pattern=r'\w+|\?|\!|\.|,|;|:',
# )
#
# # Process training data
# print("Processing training data...")
# X_train_tfidf = vectorizer.fit_transform(train_features['combined_narrative'])
# train_engineered = create_engineered_features(train_features)
# X_train = hstack([X_train_tfidf, train_engineered])
#
# # Process test data
# print("Processing test data...")
# X_test_tfidf = vectorizer.transform(test_features['combined_narrative'])
# test_engineered = create_engineered_features(test_features)
# X_test = hstack([X_test_tfidf, test_engineered])
#
#
# def balanced_sampling(X, y, n_samples=None):
#     """Perform balanced sampling with dynamic sample size"""
#     if n_samples is None:
#         minority_count = min(sum(y == 0), sum(y == 1))
#         n_samples = max(2000, minority_count * 2)
#
#     # Convert sparse matrix to dense for sampling
#     X_dense = X.toarray()
#     combined_data = pd.DataFrame(X_dense)
#     combined_data['target'] = y
#
#     # Separate classes
#     df_majority = combined_data[combined_data['target'] == 0]
#     df_minority = combined_data[combined_data['target'] == 1]
#
#     # Sampling
#     if len(df_majority) > n_samples:
#         df_majority = df_majority.sample(n=n_samples, random_state=42)
#     if len(df_minority) > n_samples:
#         df_minority = df_minority.sample(n=n_samples, random_state=42)
#     else:
#         df_minority = df_minority.sample(n=n_samples, replace=True, random_state=42)
#
#     # Combine samples
#     df_balanced = pd.concat([df_majority, df_minority])
#     y_balanced = df_balanced['target']
#     X_balanced = df_balanced.drop('target', axis=1)
#
#     return csr_matrix(X_balanced), y_balanced
#
#
# # Define columns
# binary_columns = [
#     'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt',
#     'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem',
#     'MentalHealthProblem', 'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia',
#     'DiagnosisBipolar', 'DiagnosisAdhd', 'IntimatePartnerProblem',
#     'FamilyRelationship', 'Argument', 'SchoolProblem', 'RecentCriminalLegalProblem',
#     'SuicideNote', 'SuicideIntentDisclosed', 'DisclosedToIntimatePartner',
#     'DisclosedToOtherFamilyMember', 'DisclosedToFriend'
# ]
#
# categorical_columns = ['InjuryLocationType', 'WeaponType1']
#
# # Train binary models with optimizations
# print("Training binary classification models...")
# binary_predictions = {}
#
# for col in binary_columns:
#     print(f"\nTraining model for {col}")
#     y = train_labels[col]
#
#     # Calculate optimal sample size
#     minority_count = min(sum(y == 0), sum(y == 1))
#     sample_size = max(2000, minority_count * 2)
#
#     # Balanced sampling
#     X_balanced, y_balanced = balanced_sampling(X_train, y, sample_size)
#
#     # Model parameters based on column type
#     if col.startswith('Disclosed'):
#         params = {
#             'objective': 'binary:logistic',
#             'max_depth': 8,
#             'min_child_weight': 2,
#             'subsample': 0.85,
#             'colsample_bytree': 0.85,
#             'learning_rate': 0.003,
#             'n_estimators': 500,
#             'gamma': 0.1,
#             'reg_alpha': 0.1,
#             'reg_lambda': 1,
#             'tree_method': 'hist',
#             'eval_metric': ['auc', 'logloss']
#         }
#     else:
#         params = {
#             'objective': 'binary:logistic',
#             'max_depth': 7,
#             'min_child_weight': 2,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'learning_rate': 0.005,
#             'n_estimators': 400,
#             'gamma': 0.2,
#             'reg_alpha': 0.2,
#             'reg_lambda': 1.2,
#             'tree_method': 'hist',
#             'eval_metric': ['auc', 'logloss']
#         }
#
#     # Train-validation split
#     X_train_split, X_val, y_train_split, y_val = train_test_split(
#         X_balanced, y_balanced,
#         test_size=0.2,
#         random_state=42,
#         stratify=y_balanced
#     )
#
#     # Initialize and train model
#     model = XGBClassifier(**params)
#     model.fit(
#         X_train_split,
#         y_train_split,
#         eval_set=[(X_val, y_val)],
#         verbose=False
#     )
#
#     # Make predictions
#     binary_predictions[col] = model.predict(X_test)
#     print(f"Predictions distribution: {np.unique(binary_predictions[col], return_counts=True)}")
#
#
# def train_categorical_model(X, y, model_name):
#     print(f"\nTraining {model_name}...")
#
#     # Compute class weights
#     class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
#     weight_dict = dict(zip(np.unique(y), class_weights))
#
#     params = {
#         'objective': 'multi:softmax',
#         'num_class': len(np.unique(y)),
#         'max_depth': 8,
#         'min_child_weight': 2,
#         'subsample': 0.85,
#         'colsample_bytree': 0.85,
#         'learning_rate': 0.005,
#         'n_estimators': 500,
#         'gamma': 0.1,
#         'reg_alpha': 0.1,
#         'reg_lambda': 1,
#         'tree_method': 'hist',
#         'eval_metric': ['mlogloss']
#     }
#
#     # Cross-validation
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     model = XGBClassifier(**params)
#
#     scores = []
#     predictions = np.zeros(X_test.shape[0])
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X.toarray(), y)):
#         X_train_fold = X[train_idx]
#         y_train_fold = y[train_idx]
#         X_val_fold = X[val_idx]
#         y_val_fold = y[val_idx]
#
#         sample_weights = np.array([weight_dict[cls] for cls in y_train_fold])
#
#         model.fit(
#             X_train_fold,
#             y_train_fold,
#             sample_weight=sample_weights,
#             eval_set=[(X_val_fold, y_val_fold)],
#             verbose=False
#         )
#
#         eval_results = model.evals_result_
#         fold_scores = [eval_results['validation_0']['mlogloss'][-1]]
#         scores.extend(fold_scores)
#
#         fold_predictions = model.predict(X_test)
#         predictions += fold_predictions
#
#     # Average predictions from all folds
#     predictions = np.round(predictions / 5).astype(int)
#     print(f"{model_name} average CV log loss: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")
#
#     return predictions
#
#
# # Train categorical models
# print("\nTraining categorical models...")
# injury_location_pred = train_categorical_model(
#     X_train,
#     train_labels['InjuryLocationType'] - 1,
#     'Injury Location'
# ) + 1
#
# weapon_type_pred = train_categorical_model(
#     X_train,
#     train_labels['WeaponType1'] - 1,
#     'Weapon Type'
# ) + 1
#
# # Create submission
# print("\nCreating submission file...")
# submission_format = pd.read_csv('data/submission_format.csv')
# submission = pd.DataFrame({'uid': test_features['uid']})
#
# # Add binary predictions
# for col in binary_columns:
#     submission[col] = binary_predictions[col]
#
# # Add categorical predictions
# submission['InjuryLocationType'] = injury_location_pred
# submission['WeaponType1'] = weapon_type_pred
#
# # Ensure format matches
# submission = submission_format[['uid']].merge(submission, on='uid', how='left')
# submission = submission.fillna(0)
#
# # Convert to correct data types
# for col in submission.columns:
#     if col != 'uid':
#         submission[col] = submission[col].astype(int)
#
# # Save submission
# submission.to_csv('submission.csv', index=False)
# print("\nSubmission saved successfully!")
#
# # Verify submission
# print("\nVerifying submission...")
# print("Submission shape:", submission.shape)
# print("\nColumn distributions:")
# for col in binary_columns + categorical_columns:
#     print(f"{col}: {submission[col].value_counts().to_dict()}")
#









# import numpy as np
# import pandas as pd
# from scipy.sparse import hstack, csr_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import resample
# from sklearn.utils.class_weight import compute_class_weight
# from xgboost import XGBClassifier
#
# # Load the data
# train_features = pd.read_csv('assets/train_features.csv')
# train_labels = pd.read_csv('assets/train_labels.csv')
# test_features = pd.read_csv('data/test_features.csv')
#
# # Create combined narratives first
# print("Creating combined narratives...")
# train_features['combined_narrative'] = train_features['NarrativeLE'] + ' ' + train_features['NarrativeCME']
# test_features['combined_narrative'] = test_features['NarrativeLE'] + ' ' + test_features['NarrativeCME']
#
#
# def create_engineered_features(df):
#     """Create engineered features from the narrative text and other columns"""
#     # [Previous feature engineering code remains the same]
#     features = []
#
#     # Text length features
#     features.append(df['combined_narrative'].str.len().values.reshape(-1, 1))
#     features.append(df['combined_narrative'].str.split().str.len().values.reshape(-1, 1))
#
#     # [Rest of the feature engineering code remains the same]
#     disclosure_keywords = ['told', 'said', 'mentioned', 'expressed', 'talked', 'discussed', 'confided', 'shared',
#                            'revealed']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in disclosure_keywords)
#     ).values.reshape(-1, 1))
#
#     family_keywords = ['family', 'mother', 'father', 'sister', 'brother', 'daughter', 'son',
#                        'wife', 'husband', 'spouse', 'parent', 'child', 'sibling']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in family_keywords)
#     ).values.reshape(-1, 1))
#
#     friend_keywords = ['friend', 'colleague', 'coworker', 'neighbor', 'peer', 'acquaintance',
#                        'buddy', 'companion', 'roommate', 'classmate']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in friend_keywords)
#     ).values.reshape(-1, 1))
#
#     emotion_keywords = ['depressed', 'sad', 'angry', 'upset', 'stressed', 'anxious', 'worried',
#                         'frustrated', 'hopeless', 'helpless', 'lonely', 'isolated']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in emotion_keywords)
#     ).values.reshape(-1, 1))
#
#     communication_keywords = ['text', 'call', 'phone', 'message', 'email', 'letter', 'note',
#                               'wrote', 'called', 'texted', 'messaged', 'contacted']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in communication_keywords)
#     ).values.reshape(-1, 1))
#
#     time_keywords = ['today', 'yesterday', 'week', 'month', 'recent', 'lately', 'previously']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in time_keywords)
#     ).values.reshape(-1, 1))
#
#     feature_matrix = np.hstack(features)
#     scaler = StandardScaler()
#     feature_matrix = scaler.fit_transform(feature_matrix)
#
#     return csr_matrix(feature_matrix)
#
#
# # Create TF-IDF vectors
# print("Creating TF-IDF features...")
# vectorizer = TfidfVectorizer(
#     max_features=20000,
#     stop_words='english',
#     ngram_range=(1, 3),
#     min_df=2,
#     max_df=0.95,
#     sublinear_tf=True
# )
#
# # Process training data
# print("Processing training data...")
# X_train_tfidf = vectorizer.fit_transform(train_features['combined_narrative'])
# train_engineered = create_engineered_features(train_features)
# X_train = hstack([X_train_tfidf, train_engineered])
#
# # Process test data
# print("Processing test data...")
# X_test_tfidf = vectorizer.transform(test_features['combined_narrative'])
# test_engineered = create_engineered_features(test_features)
# X_test = hstack([X_test_tfidf, test_engineered])
#
#
# def balanced_sampling(X, y, n_samples=1000):
#     """
#     Perform balanced sampling using up and down sampling
#     """
#     # Convert sparse matrix to DataFrame for sampling
#     X_dense = X.toarray()
#     combined_data = pd.DataFrame(X_dense)
#     combined_data['target'] = y
#
#     # Separate majority and minority classes
#     df_majority = combined_data[combined_data['target'] == 0]
#     df_minority = combined_data[combined_data['target'] == 1]
#
#     # Downsample majority class
#     df_majority_downsampled = resample(df_majority,
#                                        replace=False,
#                                        n_samples=n_samples,
#                                        random_state=42)
#
#     # Upsample minority class
#     df_minority_upsampled = resample(df_minority,
#                                      replace=True,
#                                      n_samples=n_samples,
#                                      random_state=42)
#
#     # Combine minority class with downsampled majority class
#     df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])
#
#     # Separate features and target
#     y_balanced = df_balanced['target']
#     X_balanced = df_balanced.drop('target', axis=1)
#
#     return csr_matrix(X_balanced), y_balanced
#
#
# # Define the columns
# binary_columns = [
#     'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt',
#     'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem',
#     'MentalHealthProblem', 'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia',
#     'DiagnosisBipolar', 'DiagnosisAdhd', 'IntimatePartnerProblem',
#     'FamilyRelationship', 'Argument', 'SchoolProblem', 'RecentCriminalLegalProblem',
#     'SuicideNote', 'SuicideIntentDisclosed', 'DisclosedToIntimatePartner',
#     'DisclosedToOtherFamilyMember', 'DisclosedToFriend'
# ]
#
# categorical_columns = ['InjuryLocationType', 'WeaponType1']
#
# # Enhanced binary classification with balanced sampling
# print("Training binary classification models with balanced sampling...")
# binary_predictions = {}
#
# for col in binary_columns:
#     print(f"\nTraining model for {col}")
#     y = train_labels[col]
#
#     # Perform balanced sampling
#     X_balanced, y_balanced = balanced_sampling(X_train, y)
#
#     # Display class distribution after sampling
#     print(f"Class distribution for {col} after sampling: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
#
#     # Calculate class weights for the balanced data
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_balanced), y=y_balanced)
#     class_weight_dict = dict(zip(np.unique(y_balanced), class_weights))
#
#     # Define model parameters
#     if col.startswith('Disclosed'):
#         params = {
#             'objective': 'binary:logistic',
#             'max_depth': 8,
#             'min_child_weight': 1,
#             'subsample': 0.9,
#             'colsample_bytree': 0.9,
#             'learning_rate': 0.005,
#             'n_estimators': 300,
#             'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],
#             'tree_method': 'hist'
#         }
#     else:
#         params = {
#             'objective': 'binary:logistic',
#             'max_depth': 6,
#             'min_child_weight': 1,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'learning_rate': 0.01,
#             'n_estimators': 200,
#             'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],
#             'tree_method': 'hist'
#         }
#
#     # Initialize and train the model
#     binary_model = XGBClassifier(**params)
#
#     # Use 3-fold cross-validation for rare classes to avoid warnings
#     cv_folds = 3 if train_labels['WeaponType1'].value_counts().min() < 5 else 5
#
#     # Cross-validation with the balanced data
#     cv_scores = cross_val_score(
#         binary_model, X_balanced, y_balanced,
#         cv=StratifiedKFold(n_splits=cv_folds),
#         scoring='f1'
#     )
#     print(f"Cross-validation F1 scores for {col}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
#
#     # Fit the model on the balanced data
#     binary_model.fit(X_balanced, y_balanced)
#
#     # Predict on the test data
#     binary_predictions[col] = binary_model.predict(X_test)
#
#
# def train_categorical_model(X, y, model_name):
#     print(f"\nTraining {model_name}...")
#
#     class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
#     weight_dict = dict(zip(np.unique(y), class_weights))
#     sample_weights = np.array([weight_dict[cls] for cls in y])
#
#     model = XGBClassifier(
#         objective='multi:softmax',
#         num_class=len(np.unique(y)),
#         max_depth=7,
#         min_child_weight=1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         learning_rate=0.01,
#         n_estimators=300,
#         tree_method='hist'
#     )
#
#     cv_scores = cross_val_score(
#         model, X, y,
#         cv=StratifiedKFold(n_splits=5),
#         scoring='f1_weighted'
#     )
#     print(f"Cross-validation F1 scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
#
#     model.fit(X, y, sample_weight=sample_weights)
#     return model.predict(X_test)
#
#
# # Train categorical models
# print("\nTraining categorical models...")
# injury_location_pred = train_categorical_model(
#     X_train,
#     train_labels['InjuryLocationType'] - 1,
#     'Injury Location'
# ) + 1
#
# weapon_type_pred = train_categorical_model(
#     X_train,
#     train_labels['WeaponType1'] - 1,
#     'Weapon Type'
# ) + 1
#
# # Process test data
# print("\nProcessing test data...")
# X_test_tfidf = vectorizer.transform(test_features['combined_narrative'])
# test_engineered = create_engineered_features(test_features)
# X_test = hstack([X_test_tfidf, test_engineered])
# print(f"Test data shape after processing: {X_test.shape}")
#
# # Modified binary classification training with additional debugging
# binary_predictions = {}
#
# for col in binary_columns:
#     print(f"\n{'=' * 50}")
#     print(f"Training model for {col}")
#     print(f"{'=' * 50}")
#
#     y = train_labels[col]
#     print(f"Original class distribution for {col}: {dict(zip(*np.unique(y, return_counts=True)))}")
#
#     # Calculate appropriate sample size based on class distribution
#     minority_count = min(sum(y == 0), sum(y == 1))
#     sample_size = max(1000, minority_count)  # Use at least 1000 samples or minority class size
#
#     # Perform balanced sampling
#     X_balanced, y_balanced = balanced_sampling(X_train, y, n_samples=sample_size)
#
#     # Define model parameters
#     params = {
#         'objective': 'binary:logistic',
#         'max_depth': 6,
#         'min_child_weight': 1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'learning_rate': 0.01,
#         'n_estimators': 200,
#         'tree_method': 'hist',
#         'eval_metric': 'logloss'  # Add eval metric for better monitoring
#     }
#
#     # Initialize and train the model
#     binary_model = XGBClassifier(**params)
#
#     # Fit the model with early stopping and eval set
#     eval_set = [(X_balanced, y_balanced)]
#     binary_model.fit(X_balanced, y_balanced,
#                      eval_set=eval_set,
#                      verbose=True)
#
#     # Make predictions
#     predictions = binary_model.predict(X_test)
#     print(f"\nPredictions distribution for {col}: {dict(zip(*np.unique(predictions, return_counts=True)))}")
#
#     # Store predictions
#     binary_predictions[col] = predictions
#
# # Create submission with verification
# submission_format = pd.read_csv('data/submission_format.csv')
# submission = pd.DataFrame({'uid': test_features['uid']})
#
# # Add predictions and verify
# for col in binary_columns:
#     if col in binary_predictions:
#         submission[col] = binary_predictions[col]
#         print(f"\nColumn {col} predictions distribution: {submission[col].value_counts().to_dict()}")
#     else:
#         print(f"Warning: No predictions found for {col}")
#         submission[col] = 0
#
# # Add categorical predictions
# submission['InjuryLocationType'] = injury_location_pred
# submission['WeaponType1'] = weapon_type_pred
#
# # Verify submission format
# print("\nVerifying submission format...")
# print(f"Submission shape: {submission.shape}")
# print(f"Expected shape from format: {submission_format.shape}")
#
# # Ensure proper column alignment
# submission = submission_format[['uid']].merge(submission, on='uid', how='left').fillna(0)
#
# # Verify no missing values
# print("\nChecking for missing values:")
# print(submission.isnull().sum())
#
# # Save submission
# submission.to_csv('submission.csv', index=False)
# print("\nSubmission saved. Verifying predictions in saved file...")
# saved_submission = pd.read_csv('submission.csv')
# print("\nPrediction distributions in saved file:")
# for col in binary_columns + categorical_columns:
#     print(f"{col}: {saved_submission[col].value_counts().to_dict()}")


















# import numpy as np
# import pandas as pd
# from imblearn.under_sampling import RandomUnderSampler
# from scipy.sparse import hstack, csr_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from xgboost import XGBClassifier
#
# # Load the data
# train_features = pd.read_csv('assets/train_features.csv')
# train_labels = pd.read_csv('assets/train_labels.csv')
# test_features = pd.read_csv('data/test_features.csv')
#
# # Create combined narratives first
# print("Creating combined narratives...")
# train_features['combined_narrative'] = train_features['NarrativeLE'] + ' ' + train_features['NarrativeCME']
# test_features['combined_narrative'] = test_features['NarrativeLE'] + ' ' + test_features['NarrativeCME']
#
#
# def create_engineered_features(df):
#     """Create engineered features from the narrative text and other columns"""
#     features = []
#
#     # Text length features
#     features.append(df['combined_narrative'].str.len().values.reshape(-1, 1))
#     features.append(df['combined_narrative'].str.split().str.len().values.reshape(-1, 1))
#
#     # Specific content indicators
#     disclosure_keywords = ['told', 'said', 'mentioned', 'expressed', 'talked', 'discussed', 'confided', 'shared',
#                            'revealed']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in disclosure_keywords)
#     ).values.reshape(-1, 1))
#
#     family_keywords = ['family', 'mother', 'father', 'sister', 'brother', 'daughter', 'son',
#                        'wife', 'husband', 'spouse', 'parent', 'child', 'sibling']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in family_keywords)
#     ).values.reshape(-1, 1))
#
#     friend_keywords = ['friend', 'colleague', 'coworker', 'neighbor', 'peer', 'acquaintance',
#                        'buddy', 'companion', 'roommate', 'classmate']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in friend_keywords)
#     ).values.reshape(-1, 1))
#
#     # Emotional content indicators
#     emotion_keywords = ['depressed', 'sad', 'angry', 'upset', 'stressed', 'anxious', 'worried',
#                         'frustrated', 'hopeless', 'helpless', 'lonely', 'isolated']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in emotion_keywords)
#     ).values.reshape(-1, 1))
#
#     # Communication method indicators
#     communication_keywords = ['text', 'call', 'phone', 'message', 'email', 'letter', 'note',
#                               'wrote', 'called', 'texted', 'messaged', 'contacted']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in communication_keywords)
#     ).values.reshape(-1, 1))
#
#     # Time indicators
#     time_keywords = ['today', 'yesterday', 'week', 'month', 'recent', 'lately', 'previously']
#     features.append(df['combined_narrative'].apply(
#         lambda x: sum(1 for word in str(x).lower().split() if word in time_keywords)
#     ).values.reshape(-1, 1))
#
#     # Stack all features horizontally
#     feature_matrix = np.hstack(features)
#
#     # Scale the features
#     scaler = StandardScaler()
#     feature_matrix = scaler.fit_transform(feature_matrix)
#
#     return csr_matrix(feature_matrix)
#
#
# # Create TF-IDF vectors
# print("Creating TF-IDF features...")
# vectorizer = TfidfVectorizer(
#     max_features=20000,
#     stop_words='english',
#     ngram_range=(1, 3),
#     min_df=2,
#     max_df=0.95,
#     sublinear_tf=True
# )
#
# # Process training data
# print("Processing training data...")
# X_train_tfidf = vectorizer.fit_transform(train_features['combined_narrative'])
# train_engineered = create_engineered_features(train_features)
# X_train = hstack([X_train_tfidf, train_engineered])
#
# # Process test data
# print("Processing test data...")
# X_test_tfidf = vectorizer.transform(test_features['combined_narrative'])
# test_engineered = create_engineered_features(test_features)
# X_test = hstack([X_test_tfidf, test_engineered])
#
# # Define the columns
# binary_columns = [
#     'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt',
#     'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem',
#     'MentalHealthProblem', 'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia',
#     'DiagnosisBipolar', 'DiagnosisAdhd', 'IntimatePartnerProblem',
#     'FamilyRelationship', 'Argument', 'SchoolProblem', 'RecentCriminalLegalProblem',
#     'SuicideNote', 'SuicideIntentDisclosed', 'DisclosedToIntimatePartner',
#     'DisclosedToOtherFamilyMember', 'DisclosedToFriend'
# ]
#
# categorical_columns = ['InjuryLocationType', 'WeaponType1']
#
#
# # Enhanced binary classification with undersampling
# print("Training binary classification models with undersampling...")
# binary_predictions = {}
#
# for col in binary_columns:
#     print(f"\nTraining model for {col}")
#     y = train_labels[col]
#
#     # Initialize the undersampler
#     undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
#     X_resampled, y_resampled = undersampler.fit_resample(X_train, y)
#
#     # Display class distribution after undersampling
#     print(f"Class distribution for {col} after undersampling: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
#
#     # Calculate class weights for the resampled data
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
#     class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))
#
#     # Define model parameters with class weights
#     if col.startswith('Disclosed'):
#         # Custom parameters for disclosure-related columns
#         params = {
#             'objective': 'binary:logistic',
#             'max_depth': 8,
#             'min_child_weight': 1,
#             'subsample': 0.9,
#             'colsample_bytree': 0.9,
#             'learning_rate': 0.005,
#             'n_estimators': 300,
#             'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],
#             'tree_method': 'hist'
#         }
#     else:
#         # Default parameters for other columns
#         params = {
#             'objective': 'binary:logistic',
#             'max_depth': 6,
#             'min_child_weight': 1,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'learning_rate': 0.01,
#             'n_estimators': 200,
#             'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],
#             'tree_method': 'hist'
#         }
#
#     # Initialize and train the model
#     binary_model = XGBClassifier(**params)
#
#     # Cross-validation with the undersampled data
#     cv_scores = cross_val_score(
#         binary_model, X_resampled, y_resampled,
#         cv=StratifiedKFold(n_splits=5),
#         scoring='f1'
#     )
#     print(
#         f"Cross-validation F1 scores for {col} after undersampling: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
#
#     # Fit the model on the resampled data
#     binary_model.fit(X_resampled, y_resampled)
#
#     # Predict on the test data
#     binary_predictions[col] = binary_model.predict(X_test)
#
#
# # Train categorical models
# def train_categorical_model(X, y, model_name):
#     print(f"\nTraining {model_name}...")
#
#     class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
#     weight_dict = dict(zip(np.unique(y), class_weights))
#     sample_weights = np.array([weight_dict[cls] for cls in y])
#
#     model = XGBClassifier(
#         objective='multi:softmax',
#         num_class=len(np.unique(y)),
#         max_depth=7,
#         min_child_weight=1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         learning_rate=0.01,
#         n_estimators=300,
#         tree_method='hist'
#     )
#
#     cv_scores = cross_val_score(
#         model, X, y,
#         cv=StratifiedKFold(n_splits=5),
#         scoring='f1_weighted'
#     )
#     print(f"Cross-validation F1 scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
#
#     model.fit(X, y, sample_weight=sample_weights)
#     return model.predict(X_test)
#
#
# # Train categorical models
# print("\nTraining categorical models...")
# injury_location_pred = train_categorical_model(
#     X_train,
#     train_labels['InjuryLocationType'] - 1,
#     'Injury Location'
# ) + 1
#
# weapon_type_pred = train_categorical_model(
#     X_train,
#     train_labels['WeaponType1'] - 1,
#     'Weapon Type'
# ) + 1
#
# # Ensure uid alignment between test features and submission format
# submission_format = pd.read_csv('data/submission_format.csv')
# submission = pd.DataFrame({'uid': test_features['uid']})
#
# # Add predictions for binary columns to the submission DataFrame
# for col in binary_columns:
#     if col in binary_predictions:
#         submission[col] = binary_predictions[col]
#     else:
#         submission[col] = 0  # Default to 0 if prediction is missing
#
# # Add categorical predictions
# submission['InjuryLocationType'] = injury_location_pred
# submission['WeaponType1'] = weapon_type_pred
#
# # Ensure all columns in the submission format are present in `submission`
# submission = submission_format[['uid']].merge(submission, on='uid', how='left')
#
# # Verify column and row alignment by ensuring each submission column exists and is in the right order
# for col in submission_format.columns:
#     if col not in submission.columns:
#         submission[col] = 0  # Add missing columns and set default values
#     else:
#         submission[col].fillna(0, inplace=True)  # Fill any NaN values with 0
#
# # Ensure correct data types
# for col in submission.columns:
#     if col != 'uid':
#         submission[col] = submission[col].astype(int)
#
# # Save the final submission
# submission.to_csv('submission.csv', index=False)
# print("\nSubmission file created successfully and verified for correct structure!")
