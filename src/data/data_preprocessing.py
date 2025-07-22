import os
import re
import numpy as np
import pandas as pd
import nltk
import logging
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)

# Download required NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {e}")
    raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logging.error(f"Lemmatization failed: {e}")
        return text

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in str(text).split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logging.error(f"Removing stop words failed: {e}")
        return text

def removing_numbers(text: str) -> str:
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error(f"Removing numbers failed: {e}")
        return text

def lower_case(text: str) -> str:
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logging.error(f"Lowercasing failed: {e}")
        return text

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        return " ".join(text.split()).strip()
    except Exception as e:
        logging.error(f"Removing punctuations failed: {e}")
        return text

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Removing URLs failed: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
        return df
    except Exception as e:
        logging.error(f"Removing small sentences failed: {e}")
        return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].astype(str)
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error(f"Text normalization failed: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Sentence normalization failed: {e}")
        return sentence

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved data to {path}")
    except Exception as e:
        logging.error(f"Failed to save data to {path}: {e}")
        raise

def main() -> None:
    try:
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")

        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Data preprocessing pipeline failed: {e}")

if __name__ == "__main__":
    main()