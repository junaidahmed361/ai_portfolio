import pandas as pd
import nltk
import sys
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Read in the data
def read_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    return df

# perform tokenization
def tokenize(col: pd.Series) -> pd.Series:
    print("--Tokenizing...")
    return col.apply(lambda x: word_tokenize(x))

# perform lemmatization
def lemmatize(col: pd.Series) -> pd.Series:
    print("--Lemmatizing...")
    lemmatizer = WordNetLemmatizer()
    return col.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# perform stemming
def stem(col: pd.Series) -> pd.Series:
    print("--Stemming...")
    stemmer = PorterStemmer()
    return col.apply(lambda x: [stemmer.stem(word) for word in x])

# perform all three steps
def preprocess(col: pd.Series) -> pd.Series:
    print("Preprocessing...")
    return stem(lemmatize(tokenize(col)))

# main function
def main():
    # use command args to pass in the file name
    args = sys.argv[1:]
    file_name = args[0]
    print(file_name)

    df = read_data(file_name)
    df['summary'] = preprocess(df['summary'])
    print(df['summary'].head())

if __name__ == '__main__':
    main()
