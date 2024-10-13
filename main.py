import os
import nltk
from fuzzywuzzy import fuzz
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd

# Initialize stopwords and lemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters,
    tokenizing, removing stop words, and lemmatizing.

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Preprocessed text
    """
    try:
        # Convert to lowercase and handle non-string input
        text = str(text).lower()

        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words and lemmatize
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        return ' '.join(processed_tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""


# Fuzzy matching function
def fuzzy_match_keywords(processed_narrative, keywords_map, threshold=80):
    """Fuzzily match keywords in the processed text based on similarity score."""
    result = {col: 0 for col in keywords_map}  # Initialize all columns to 0
    for col, keywords in keywords_map.items():
        for keyword in keywords:
            # Check similarity between each word in the text and the keyword
            for word in processed_narrative.split():
                if fuzz.ratio(word, keyword) >= threshold:
                    result[col] = 1
                    break  # No need to check more keywords if one is found
    return result


# Clear files if they exist and have data
def clear_file_if_not_empty(file_path):
    """Clear a file if it's not empty."""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        open(file_path, 'w').close()
        print(f"Cleared file: {file_path}")


# Load CSV data
def load_data(features_file):
    """Load feature and label data from CSV files."""
    try:
        features_df = pd.read_csv(features_file)
        return features_df
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        raise


# Main execution block
if __name__ == "__main__":
    try:
        # Clear the submission file if necessary
        clear_file_if_not_empty('assets/processed_data.csv')

        # Load the data
        features_df = load_data('data/smoke_test_features.csv')

        # Combine and preprocess NarrativeLE and NarrativeCME
        print("Starting text preprocessing...")

        # Combine the narratives and apply preprocessing
        features_df['processed_narrative'] = (features_df['NarrativeLE'].fillna('') + ' ' +
                                              features_df['NarrativeCME'].fillna('')).astype(str)

        # Apply preprocessing to the combined narrative column
        features_df['processed_narrative'] = features_df['processed_narrative'].apply(preprocess_text)

        # Print the first few rows of uid and processed_narrative
        print(features_df[['uid', 'processed_narrative']].head())

        # Save the processed DataFrame to CSV
        features_df.to_csv('assets/processed_data.csv', index=False)
        print("Processed data saved to 'assets/processed_data.csv'.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Dictionary mapping columns to keywords/phrases to detect in the text
keywords_map = {
    'DepressedMood': ['depressed', 'sad', 'hopeless', 'down', 'low', 'blue', 'feeling empty', 'miserable'],
    'MentalIllnessTreatmentCurrnt': ['therapy', 'counseling', 'treatment', 'medication', 'psychotherapy', 'CBT', 'DBT',
                                     'psychiatry', 'group therapy', 'antidepressants', 'mood stabilizers'],
    'HistoryMentalIllnessTreatmnt': ['history of treatment', 'previous treatment', 'past counseling',
                                     'prior medication', 'former therapy', 'previous psychiatric care'],
    'SuicideAttemptHistory': ['attempted suicide', 'past suicide attempt', 'previous self-harm', 'prior attempt',
                              'failed suicide attempt'],
    'SuicideThoughtHistory': ['suicidal thoughts', 'suicide ideation', 'thoughts of ending life', 'considered suicide',
                              'death wish'],
    'SubstanceAbuseProblem': ['substance abuse', 'alcohol problem', 'drug addiction', 'drug abuse', 'alcoholism',
                              'substance dependence', 'binge drinking', 'opioid addiction'],
    'MentalHealthProblem': ['mental illness', 'mental health issue', 'psychological disorder', 'emotional instability',
                            'psychiatric problem', 'emotional distress'],
    'DiagnosisAnxiety': ['anxiety', 'panic attacks', 'generalized anxiety', 'social anxiety', 'GAD', 'nervousness',
                         'excessive worry'],
    'DiagnosisDepressionDysthymia': ['depression', 'dysthymia', 'major depressive disorder',
                                     'persistent depressive disorder', 'clinical depression', 'chronic depression'],
    'DiagnosisBipolar': ['manic depression', 'bipolar', 'cyclothymia', 'manic episodes', 'depressive episodes'],
    'DiagnosisAdhd': ['adhd', 'attention deficit', 'hyperactivity', 'ADD', 'attention deficit hyperactivity disorder',
                      'impulsivity', 'difficulty focusing'],
    'IntimatePartnerProblem': ['partner problem', 'relationship issue', 'marital conflict', 'spousal issue',
                               'domestic dispute', 'relationship breakdown', 'romantic partner conflict'],
    'FamilyRelationship': ['family problem', 'family issue', 'parental conflict', 'sibling rivalry', 'family tension',
                           'home conflict', 'family stress'],
    'Argument': ['argument', 'conflict', 'disagreement', 'quarrel', 'fight', 'dispute', 'verbal altercation',
                 'heated exchange'],
    'SchoolProblem': ['school issue', 'academic problem', 'bullying', 'truancy', 'failing grades', 'disciplinary issue',
                      'peer conflict', 'learning difficulties'],
    'RecentCriminalLegalProblem': ['legal problem', 'criminal charges', 'arrest', 'court case', 'probation',
                                   'criminal investigation', 'pending trial', 'incarceration'],
    'SuicideNote': ['suicide note', 'left note', 'final message', 'goodbye letter', 'written note', 'suicide letter',
                    'farewell note'],
    'SuicideIntentDisclosed': ['disclosed intent', 'told someone', 'revealed suicide plan', 'shared intent',
                               'mentioned suicide', 'confessed suicidal thoughts'],
    'DisclosedToIntimatePartner': ['told partner', 'disclosed to spouse', 'told boyfriend', 'told girlfriend',
                                   'confided in partner', 'revealed to partner'],
    'DisclosedToOtherFamilyMember': ['told family', 'disclosed to sibling', 'told parents', 'told child',
                                     'confided in family member', 'revealed to relative'],
    'DisclosedToFriend': ['told friend', 'disclosed to friend', 'confided in friend', 'revealed to close friend',
                          'shared with friend']

}

# Mapping keywords to integers for categorical variables
categorical_mappings = {
    'InjuryLocationType': {
        1: ['house', 'apartment', 'condo', 'townhouse', 'cabin', 'bungalow', 'duplex', 'villa', 'studio'],
        2: ['motor', 'car', 'truck', 'SUV', 'motorcycle', 'van', 'RV', 'scooter',
            'ride', 'bicycle', 'vehicle'],
        3: ['nature', 'field', 'river', 'beach', 'woods', 'forest', 'mountain', 'lake', 'meadow', 'trail',
            'desert', 'canyon', 'waterfall', 'creek'],
        4: ['park', 'playground', 'public', 'park', 'field', 'amusement', 'zoo', 'plaza',
            'square', 'outdoor', 'pool', 'garden'],
        5: ['street', 'road', 'sidewalk', 'alley', 'highway', 'crosswalk', 'intersection', 'parking', 'driveway',
            'boulevard', 'avenue', 'bridge', 'overpass'],
        6: ['school', 'office', 'mall', 'airport', 'station', 'gym', 'hospital', 'restaurant',
            'church', 'theater', 'site']
    },

    'WeaponType1': {
        1: ['blunt instrument', 'bat', 'hammer', 'club', 'crowbar', 'wrench', 'brick', 'rock', 'pipe'],
        2: ['drowning', 'pool', 'lake', 'ocean', 'bathtub', 'river', 'pond'],
        3: ['fall', 'from stairs', 'from balcony', 'from ladder', 'off a roof', 'slip on ice',
            'trip on uneven surface'],
        4: ['fire', 'burn', 'flame', 'hot liquid', 'steam', 'chemical burn', 'scalding water', 'hot metal'],
        5: ['firearm', 'gun', 'pistol', 'rifle', 'shotgun', 'revolver', 'machine gun'],
        6: ['hanging', 'strangulation', 'suffocation', 'ligature', 'rope', 'belt', 'plastic bag', 'wire'],
        7: ['motor vehicle', 'bus', 'motorcycle', 'car', 'truck', 'SUV', 'van', 'tractor', 'ATV'],
        8: ['other transport vehicle', 'train', 'plane', 'boat', 'helicopter', 'subway', 'tram', 'ferry', 'jet ski'],
        9: ['poisoning', 'drug overdose', 'carbon monoxide', 'alcohol poisoning', 'toxic gas', 'chemical ingestion',
            'food poisoning'],
        10: ['sharp instrument', 'knife', 'scissors', 'axe', 'sword', 'glass shard', 'razor blade', 'box cutter'],
        11: ['other', 'taser', 'electrocution', 'nail gun', 'explosive device', 'acid attack', 'bow and arrow',
             'slingshot'],
        12: ['unknown']
    }

}


def detect_keywords_in_text(processed_narrative, keywords_map):
    """
    Detects keywords in the processed text and sets corresponding columns to 1 or 0.

    Args:
        processed_narrative (str): The preprocessed narrative text.
        keywords_map (dict): A dictionary mapping column names to keywords/phrases.

    Returns:
        dict: A dictionary where keys are column names and values are 0 or 1 (as integers).
    """
    result = {col: 0 for col in keywords_map}  # Initialize all columns to 0
    for col, keywords in keywords_map.items():
        for keyword in keywords:
            if keyword in processed_narrative:
                result[col] = 1
                break  # No need to check more keywords if one is found
    return {col: int(val) for col, val in result.items()}  # Ensure all results are integers


# Function for categorizing injury location and weapon type
def categorize_injury_location_and_weapon(processed_narrative, categorical_mappings):
    """
    Categorize InjuryLocationType and WeaponType1 based on keywords found in the processed narrative.

    Args:
        processed_narrative (str): The preprocessed narrative text.
        categorical_mappings (dict): Mapping of categories to keywords.

    Returns:
        dict: A dictionary with InjuryLocationType and WeaponType1 values as integers.
    """
    location_type = 6  # Default to 'Other'
    weapon_type = 12  # Default to 'Unknown'

    # Check for InjuryLocationType
    for code, keywords in categorical_mappings['InjuryLocationType'].items():
        if any(keyword in processed_narrative for keyword in keywords):
            location_type = code
            break  # Found a match, exit loop

    # Check for WeaponType1
    for code, keywords in categorical_mappings['WeaponType1'].items():
        if any(keyword in processed_narrative for keyword in keywords):
            weapon_type = code
            break  # Found a match, exit loop

    return {'InjuryLocationType': int(location_type), 'WeaponType1': int(weapon_type)}


def prepare_submission_with_fuzzy_matching(processed_data_path, submission_format_path='submission.csv'):
    """Prepare processed data by detecting fuzzily matched keywords."""
    try:
        # Clear submission.csv if not empty
        clear_file_if_not_empty(submission_format_path)

        # Load the processed data
        print("Loading processed data file...")
        processed_df = pd.read_csv(processed_data_path)

        # Start submission dataframe with UID
        submission_df = processed_df[['uid']].copy()

        # Process each row and detect fuzzy keywords in the text
        for idx, row in processed_df.iterrows():
            processed_narrative = row['processed_narrative']
            keyword_results = fuzzy_match_keywords(processed_narrative, keywords_map)

            # Update submission DataFrame with keyword results
            for col, value in keyword_results.items():
                submission_df.loc[idx, col] = int(value)  # Ensure integer results

            # Process each row and categorize injury location and weapon
            location_and_weapon = categorize_injury_location_and_weapon(processed_narrative, categorical_mappings)

            # Ensure the results are integers before assignment
            submission_df.loc[idx, 'InjuryLocationType'] = int(location_and_weapon['InjuryLocationType'])
            submission_df.loc[idx, 'WeaponType1'] = int(location_and_weapon['WeaponType1'])

        # Fill NaNs with 0 if necessary
        submission_df.fillna(0, inplace=True)

        # Convert all numeric columns to integers
        numeric_cols = submission_df.select_dtypes(include=['float64']).columns
        submission_df[numeric_cols] = submission_df[numeric_cols].astype(int)

        # Print data types for verification
        print("Data types after conversion:")
        print(submission_df.dtypes)

        # Save the submission DataFrame to submission.csv
        submission_df.to_csv(submission_format_path, index=False)
        print(f"Submission saved to {submission_format_path}")

        return submission_df

    except Exception as e:
        print(f"Error preparing submission: {e}")
        raise


# Example usage
if __name__ == "__main__":
    try:
        submission_df = prepare_submission_with_fuzzy_matching('assets/processed_data.csv')
        print(submission_df.head())  # Display the first few rows of the submission

    except Exception as e:
        print(f"An error occurred: {e}")
