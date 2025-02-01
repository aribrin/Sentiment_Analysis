import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('twitter_sentiment_analysis.csv', encoding='latin1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Create a subset (10% of the data)
subset_df = df.sample(frac=0.1, random_state=42)

# clean and preprocess the text data
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # remove special characters and numbers
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    # convert to lowercase
    text = text.lower()
    # tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Preprocess the subset
subset_df['cleaned_text'] = subset_df['text'].apply(preprocess_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=1000)  # Use fewer features
X_subset = tfidf.fit_transform(subset_df['cleaned_text']).toarray()
y_subset = subset_df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))