import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer




def clean_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

def tokenize(text):
    return word_tokenize(text)

def stem_sentence(sentence):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in sentence.split()])

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in sentence.split()])

def downsample_to_number(df, class_column, n_samples, random_state=42):
    downsampled_dfs = []
    for class_value in df[class_column].unique():
        df_class = df[df[class_column] == class_value]
        df_class_downsampled = df_class.sample(
            n=min(n_samples, len(df_class)), 
            random_state=random_state
        )
        downsampled_dfs.append(df_class_downsampled)
    return pd.concat(downsampled_dfs)