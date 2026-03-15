import pandas as pd
import re
import string
import spacy
from nltk.stem import WordNetLemmatizer

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Stopword removal
nlp = spacy.blank("en")

# Parse transcript using regular expressions
def parse_transcript(transcript):
    # Find all caps title, colon separator, and then find the associated content
    pattern = r'([A-Z ]+):\s*(.*?)(?=\s*[A-Z ]+:|$)'
    parts = re.findall(pattern, transcript)

    # Create a dictionary with the results
    result = {title.strip(): content.strip() for title, content in parts}

    # Find text before first title
    # Find when the first title appears
    first = re.search(r'[A-Z ]+:', transcript)

    # Extract the text before
    if first and first.start() > 0:
        prefix = transcript[:first.start()].strip()
        if prefix:
            result["UNLABELED"] = prefix

    # If there are no titles
    if not parts and transcript.strip():
        result["UNLABELED"] = transcript.strip()

    return result

# Send text to lowercase, strip whitespace, remove punctuation
def clean(text):
    text = text.lower()

    # string.punctuation without the period and hyphen
    punct = string.punctuation.replace('.', '')
    punct = punct.replace('-', '')

    # Remove punctuation
    text = text.translate(str.maketrans('', '', punct))

    # remove periods not adjacent to digits (for lists and decimals)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # Remove multiple spaces and unnecessary whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# remove stopwords
def remove_stopwords(text):
    processed = nlp(text)
    return " ".join(t.text for t in processed if not t.is_stop)

# Clean transcript dictionary
def clean_transcript(transcript_dict):
    return {k: clean(v) for k, v in transcript_dict.items()}

# Check if the transcript has any content other than its titles
def has_content(transcript_dict):
    return any(v.strip() for v in transcript_dict.values())

# Remove parts that have no content
def remove_empty_parts(transcript_dict):
    return {k : v for k, v in transcript_dict.items() if v.strip()}

# Lemmatize each word
def lemmatize(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

# Remove lone hyphens and fix whitespace
def fix_hyphens(text):
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Join the transcript dictionary
def join_sections(transcript_dict):
    return " ".join(f"{k} {v}" for k, v in transcript_dict.items())

# Apply simple tokenization (split)
def tokenize(text):
    return text.split()

# Read the dataset contents
df = pd.read_csv("mtsamples.csv")

# Add columns
df.columns = ["id", "description", "specialty", "sample_name", "transcription", "keywords"]

# Separate transcript content by tag
df["parsed_transcript"] = df["transcription"].apply(parse_transcript)

# Clean the contents of the transcript content
df["parsed_transcript"] = df["parsed_transcript"].apply(clean_transcript)

# Remove rows with no useful content
df = df[df["parsed_transcript"].apply(has_content)]

# Remove empty parts
df["parsed_transcript"] = df["parsed_transcript"].apply(remove_empty_parts)

# Join the sections, and apply the rest of the text preprocessing steps
df["parsed_transcript"] = df["parsed_transcript"].apply(join_sections)
df["parsed_transcript"] = df["parsed_transcript"].apply(clean)
df["parsed_transcript"] = df["parsed_transcript"].apply(remove_stopwords)
df["parsed_transcript"] = df["parsed_transcript"].apply(clean)
df["parsed_transcript"] = df["parsed_transcript"].apply(lemmatize)
df["parsed_transcript"] = df["parsed_transcript"].apply(fix_hyphens)
df["parsed_transcript"] = df["parsed_transcript"].apply(tokenize)

# Add length column
df["length"] = df["parsed_transcript"].apply(len)

# Remove length outliers
mean = df["length"].mean()
std = df["length"].std()

# Remove transcripts that are differ by more than 3 SD's in length
lower = mean - 3 * std
upper = mean + 3 * std
df = df[(df["length"] >= lower) & (df["length"] <= upper)]

# Export to data1.csv
df.to_csv("data1.csv", index=False)