import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load stopwords for English and initialize the lemmatizer
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
        # Load the data
        features_df = load_data('data/train_features.csv')

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
    'DepressedMood': ['depressed', 'sad', 'hopeless', 'down'],
    'MentalIllnessTreatmentCurrnt': ['therapy', 'counseling', 'treatment', 'medication'],
    'HistoryMentalIllnessTreatmnt': ['history of treatment', 'previous treatment'],
    'SuicideAttemptHistory': ['attempted suicide', 'past suicide attempt'],
    'SuicideThoughtHistory': ['suicidal thoughts', 'suicide ideation'],
    'SubstanceAbuseProblem': ['substance abuse', 'alcohol problem', 'drug addiction'],
    'MentalHealthProblem': ['mental illness', 'mental health issue'],
    'DiagnosisAnxiety': ['anxiety', 'panic attacks'],
    'DiagnosisDepressionDysthymia': ['depression', 'dysthymia'],
    'DiagnosisBipolar': ['bipolar disorder'],
    'DiagnosisAdhd': ['adhd', 'attention deficit'],
    'IntimatePartnerProblem': ['partner problem', 'relationship issue'],
    'FamilyRelationship': ['family problem', 'family issue'],
    'Argument': ['argument', 'conflict'],
    'SchoolProblem': ['school issue', 'academic problem'],
    'RecentCriminalLegalProblem': ['legal problem', 'criminal charges'],
    'SuicideNote': ['suicide note', 'left note'],
    'SuicideIntentDisclosed': ['disclosed intent', 'told someone'],
    'DisclosedToIntimatePartner': ['told partner', 'disclosed to spouse'],
    'DisclosedToOtherFamilyMember': ['told family', 'disclosed to sibling'],
    'DisclosedToFriend': ['told friend', 'disclosed to friend']
}

# Mapping keywords to integers for categorical variables
categorical_mappings = {
    'InjuryLocationType': {
        1: ['house', 'apartment'],
        2: ['motor vehicle', 'car', 'truck'],
        3: ['natural area', 'field', 'river', 'beach', 'woods'],
        4: ['park', 'playground', 'public area'],
        5: ['street', 'road', 'sidewalk', 'alley'],
        6: ['other']
    },
    'WeaponType1': {
        1: ['blunt instrument'],
        2: ['drowning'],
        3: ['fall'],
        4: ['fire', 'burn'],
        5: ['firearm', 'gun'],
        6: ['hanging', 'strangulation', 'suffocation'],
        7: ['motor vehicle', 'bus', 'motorcycle'],
        8: ['other transport vehicle', 'train', 'plane', 'boat'],
        9: ['poisoning'],
        10: ['sharp instrument'],
        11: ['other', 'taser', 'electrocution', 'nail gun'],
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
        dict: A dictionary where keys are column names and values are 0 or 1.
    """
    result = {col: 0 for col in keywords_map}  # Initialize all columns to 0
    for col, keywords in keywords_map.items():
        for keyword in keywords:
            if keyword in processed_narrative:
                result[col] = 1
                break  # No need to check more keywords if one is found
    return result


def categorize_injury_location_and_weapon(processed_narrative):
    """
    Categorize InjuryLocationType and WeaponType1 based on keywords found in the processed narrative.

    Args:
        processed_narrative (str): The preprocessed narrative text.

    Returns:
        dict: A dictionary with InjuryLocationType and WeaponType1 values.
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

    return {'InjuryLocationType': location_type, 'WeaponType1': weapon_type}


def prepare_submission_with_keywords(processed_data_path, submission_format_path='data/submission_format.csv'):
    """
    Prepare processed data according to the required submission format by detecting keywords in text.

    Args:
        processed_data_path (str): Path to the processed data CSV.
        submission_format_path (str): Path to the submission format CSV for reference (optional).

    Returns:
        pd.DataFrame: Submission-ready dataframe.
    """
    try:
        # Load the processed data
        print("Loading processed data file...")
        processed_df = pd.read_csv(processed_data_path)

        print(f"Processed data shape: {processed_df.shape}")

        # Check for NA values in processed data
        print("\nNA values in processed data:")
        print(processed_df.isna().sum())

        # Start submission dataframe with UID
        submission_df = processed_df[['uid']].copy()

        # Process each row and detect keywords in the text
        print("Detecting keywords in narratives...")
        for idx, row in processed_df.iterrows():
            processed_narrative = row['processed_narrative']
            keyword_results = detect_keywords_in_text(processed_narrative, keywords_map)

            # Update submission DataFrame with keyword results
            for col, value in keyword_results.items():
                submission_df.loc[idx, col] = value  # Set the result in the submission dataframe

            # Categorize InjuryLocationType and WeaponType1
            location_and_weapon = categorize_injury_location_and_weapon(processed_narrative)
            submission_df.loc[idx, 'InjuryLocationType'] = location_and_weapon['InjuryLocationType']
            submission_df.loc[idx, 'WeaponType1'] = location_and_weapon['WeaponType1']

        # Final verification for NA values
        print("\nFinal check for any remaining NA values:")
        print(submission_df.isna().sum())

        return submission_df

    except Exception as e:
        print(f"Error preparing submission: {e}")
        raise


# Example usage (main execution block)
if __name__ == "__main__":
    try:
        submission_df = prepare_submission_with_keywords('assets/processed_data.csv')
        print(submission_df.head())  # Display the first few rows of the submission
    except Exception as e:
        print(f"An error occurred: {e}")
