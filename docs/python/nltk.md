# NLTK (Natural Language Toolkit)

NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

## Installation

```bash
# Basic installation
pip install nltk

# Install with datasets
pip install nltk[all]

# Download specific data
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('stopwords')"
python -c "import nltk; nltk.download('vader_lexicon')"
python -c "import nltk; nltk.download('wordnet')"
python -c "import nltk; nltk.download('omw-1.4')"

# Download all datasets (large)
python -c "import nltk; nltk.download('all')"
```

## Basic Setup

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
```

## Core Functionality

### Text Tokenization

```python
# Sentence tokenization
text = "Hello world. This is NLTK. It's great for NLP!"
sentences = sent_tokenize(text)
print(sentences)  # ['Hello world.', 'This is NLTK.', "It's great for NLP!"]

# Word tokenization
words = word_tokenize(text)
print(words)  # ['Hello', 'world', '.', 'This', 'is', 'NLTK', '.', ...]

# Custom tokenizers
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer, LineTokenizer

# Only alphabetic tokens
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)

# Whitespace tokenization
ws_tokenizer = WhitespaceTokenizer()
tokens = ws_tokenizer.tokenize(text)
```

### Stop Words Removal

```python
from nltk.corpus import stopwords

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Filter stop words
words = word_tokenize("This is a sample sentence with stop words.")
filtered_words = [w for w in words if w.lower() not in stop_words]
print(filtered_words)  # ['sample', 'sentence', 'stop', 'words', '.']

# Custom stop words
custom_stops = stop_words.union({'sample', 'example'})
```

### Stemming and Lemmatization

```python
# Porter Stemmer
stemmer = PorterStemmer()
words = ["running", "runs", "ran", "runner"]
stems = [stemmer.stem(word) for word in words]
print(stems)  # ['run', 'run', 'ran', 'runner']

# WordNet Lemmatizer (more accurate)
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, pos='v') for word in words]
print(lemmas)  # ['run', 'run', 'run', 'runner']

# Lemmatize with different POS tags
word = "better"
print(lemmatizer.lemmatize(word, pos='a'))  # good (adjective)
print(lemmatizer.lemmatize(word, pos='r'))  # well (adverb)
```

### Part-of-Speech Tagging

```python
# POS tagging
text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
print(pos_tags)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ...]

# Extract specific POS
nouns = [word for word, pos in pos_tags if pos.startswith('N')]
adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]

# Universal POS tags
from nltk.tag import pos_tag
from nltk.corpus import brown
universal_tags = pos_tag(tokens, tagset='universal')
```

### Named Entity Recognition

```python
# Named entity chunking
tokens = word_tokenize("Barack Obama was the 44th President of the United States.")
pos_tags = pos_tag(tokens)
entities = ne_chunk(pos_tags)

# Extract named entities
from nltk import Tree
def extract_entities(tree):
    entities = []
    if hasattr(tree, 'label'):
        entities.append((tree.label(), [token for token, pos in tree.leaves()]))
    else:
        for child in tree:
            entities.extend(extract_entities(child))
    return entities

named_entities = extract_entities(entities)
print(named_entities)  # [('PERSON', ['Barack', 'Obama']), ...]
```

## Common Use Cases

### Sentiment Analysis

```python
# VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

texts = [
    "I love this product! It's amazing!",
    "This is terrible. I hate it.",
    "It's okay, nothing special.",
    "Best purchase ever! Highly recommend!"
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Positive: {scores['pos']:.3f}")
    print(f"Negative: {scores['neg']:.3f}")
    print(f"Neutral: {scores['neu']:.3f}")
    print(f"Compound: {scores['compound']:.3f}")
    print("-" * 50)

# Simple sentiment classification
def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
```

### Text Preprocessing Pipeline

```python
import re
import string

def preprocess_text(text, 
                   lowercase=True,
                   remove_punctuation=True,
                   remove_stopwords=True,
                   lemmatize=True):
    """Complete text preprocessing pipeline"""
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs, emails, mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# Usage
text = "I'm loving this new product! Check out https://example.com #awesome @company"
processed = preprocess_text(text)
print(processed)  # ['loving', 'new', 'product', 'check']
```

### Frequency Analysis

```python
from nltk import FreqDist
from collections import Counter

# Word frequency distribution
text = "the quick brown fox jumps over the lazy dog the fox is quick"
tokens = word_tokenize(text.lower())
fdist = FreqDist(tokens)

# Most common words
print(fdist.most_common(5))  # [('the', 3), ('quick', 2), ('fox', 2), ...]

# Plot frequency distribution
fdist.plot(30, cumulative=False)

# Conditional frequency distribution
from nltk import ConditionalFreqDist
from nltk.corpus import brown

# Frequency by genre
cfdist = ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

# Words most common in news vs romance
cfdist['news'].most_common(10)
cfdist['romance'].most_common(10)
```

### N-grams and Collocations

```python
from nltk import ngrams, collocations
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

# Generate n-grams
text = "the quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)

# Bigrams
bigrams = list(ngrams(tokens, 2))
print(bigrams[:5])  # [('the', 'quick'), ('quick', 'brown'), ...]

# Trigrams
trigrams = list(ngrams(tokens, 3))
print(trigrams[:3])  # [('the', 'quick', 'brown'), ...]

# Find collocations
from nltk.corpus import text1  # Moby Dick

# Bigram collocations
bigram_measures = collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(text1.tokens)
finder.apply_freq_filter(3)  # Only bigrams that appear 3+ times

# Best collocations by PMI
collocations = finder.nbest(bigram_measures.pmi, 10)
print(collocations)  # [('Sperm', 'Whale'), ('Moby', 'Dick'), ...]
```

## Advanced Features

### Custom Text Classification

```python
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize

# Prepare movie review dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Feature extraction
def document_features(document):
    """Extract features from document"""
    words = set(document)
    features = {}
    
    # Word presence features
    for word in word_features:
        features[f'contains({word})'] = (word in words)
    
    return features

# Get most informative words
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

# Create feature sets
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

# Train classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy:.3f}")

# Show most informative features
classifier.show_most_informative_features(5)
```

### Working with Corpora

```python
from nltk.corpus import gutenberg, reuters, wordnet

# Gutenberg corpus
print(gutenberg.fileids())  # List of books
emma = gutenberg.words('austen-emma.txt')
print(f"Emma has {len(emma)} words")

# Reuters corpus
print(reuters.categories())  # News categories
finance_docs = reuters.fileids('money-fx')
print(f"Finance articles: {len(finance_docs)}")

# WordNet (semantic dictionary)
from nltk.corpus import wordnet as wn

# Synsets (synonym sets)
dog_synsets = wn.synsets('dog')
print(dog_synsets[0].definition())  # 'a member of the genus Canis...'

# Hypernyms and hyponyms
dog = wn.synset('dog.n.01')
print(dog.hypernyms())  # More general terms
print(dog.hyponyms())   # More specific terms

# Semantic similarity
dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
similarity = dog.path_similarity(cat)
print(f"Dog-cat similarity: {similarity:.3f}")
```

### Text Parsing and Chunking

```python
# Grammar-based chunking
grammar = r"""
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    PP: {<IN><NP>}               # Chunk prepositions followed by NP
    VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs followed by NP or PP
"""

chunk_parser = nltk.RegexpParser(grammar)
sentence = "The little yellow dog barked at the cat"
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)
parsed = chunk_parser.parse(pos_tags)

# Draw parse tree
parsed.draw()

# Extract noun phrases
def extract_noun_phrases(tree):
    noun_phrases = []
    for subtree in tree:
        if hasattr(subtree, 'label') and subtree.label() == 'NP':
            np = ' '.join([token for token, pos in subtree.leaves()])
            noun_phrases.append(np)
    return noun_phrases

nps = extract_noun_phrases(parsed)
print(nps)  # ['The little yellow dog', 'the cat']
```

## Integration with Other Libraries

### With Pandas for Data Analysis

```python
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Create sample dataset
data = {
    'review': [
        "This product is amazing! Love it!",
        "Terrible quality. Very disappointed.",
        "It's okay, nothing special.",
        "Best purchase I've ever made!"
    ],
    'rating': [5, 1, 3, 5]
}

df = pd.DataFrame(data)

# Add sentiment analysis
sia = SentimentIntensityAnalyzer()
df['sentiment_compound'] = df['review'].apply(
    lambda x: sia.polarity_scores(x)['compound']
)

# Add preprocessing
def preprocess_for_analysis(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['review'].apply(preprocess_for_analysis)
df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

print(df[['review', 'sentiment_compound', 'word_count']])
```

### With Scikit-learn for ML

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Custom tokenizer using NLTK
def nltk_tokenizer(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=nltk_tokenizer, stop_words='english')),
    ('classifier', MultinomialNB())
])

# Train model (using movie reviews data)
X = [' '.join(d) for d, c in documents]
y = [c for d, c in documents]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(classification_report(y_test, predictions))
```

## Best Practices

### Performance Tips

```python
# 1. Cache expensive operations
import functools

@functools.lru_cache(maxsize=1000)
def cached_lemmatize(word, pos='n'):
    return lemmatizer.lemmatize(word, pos=pos)

# 2. Use generators for large datasets
def process_large_corpus(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield preprocess_text(line.strip())

# 3. Batch processing
def batch_sentiment_analysis(texts, batch_size=100):
    sia = SentimentIntensityAnalyzer()
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = [sia.polarity_scores(text) for text in batch]
        results.extend(batch_results)
    
    return results

# 4. Efficient stopword removal
stop_words = set(stopwords.words('english'))  # Create once, reuse many times

def remove_stopwords_efficiently(tokens):
    return [token for token in tokens if token.lower() not in stop_words]
```

### Memory Management

```python
# For large text processing
import gc
from collections import deque

def process_large_text_stream(text_stream, window_size=1000):
    """Process large text streams efficiently"""
    buffer = deque(maxlen=window_size)
    
    for text in text_stream:
        # Process text
        processed = preprocess_text(text)
        buffer.append(processed)
        
        # Periodic cleanup
        if len(buffer) == window_size:
            # Do something with buffer
            yield list(buffer)
            gc.collect()  # Force garbage collection
```

### Error Handling

```python
def robust_text_processing(text):
    """Text processing with error handling"""
    try:
        # Validate input
        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return []
        
        # Process with fallbacks
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split
            tokens = text.split()
        
        # Safe POS tagging
        try:
            pos_tags = pos_tag(tokens)
        except:
            pos_tags = [(token, 'NN') for token in tokens]
        
        return pos_tags
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return []
```

## Real-world Examples

### Complete Sentiment Analysis Pipeline

```python
class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """Clean and preprocess text"""
        # Basic cleaning
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization and normalization
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze(self, text):
        """Perform sentiment analysis"""
        # Preprocess
        clean_text = self.preprocess(text)
        
        # Get sentiment scores
        scores = self.sia.polarity_scores(text)  # Use original text for better accuracy
        
        # Classify sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(compound),
            'scores': scores,
            'processed_text': clean_text
        }

# Usage
analyzer = SentimentAnalyzer()
result = analyzer.analyze("I absolutely love this new product! It's fantastic!")
print(result)
```

### Text Summarization with NLTK

```python
from nltk.tokenize import sent_tokenize
from collections import Counter
import math

def extractive_summarization(text, num_sentences=3):
    """Simple extractive summarization using TF-IDF"""
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text
    
    # Tokenize and preprocess
    all_words = []
    sentence_words = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
        sentence_words.append(words)
        all_words.extend(words)
    
    # Calculate word frequencies
    word_freq = Counter(all_words)
    
    # Calculate sentence scores
    sentence_scores = []
    for words in sentence_words:
        score = sum(word_freq[word] for word in words)
        sentence_scores.append(score)
    
    # Get top sentences
    top_indices = sorted(range(len(sentence_scores)), 
                        key=lambda i: sentence_scores[i], 
                        reverse=True)[:num_sentences]
    
    # Return sentences in original order
    top_indices.sort()
    summary_sentences = [sentences[i] for i in top_indices]
    
    return ' '.join(summary_sentences)

# Usage
long_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
concerned with the interactions between computers and human language. In particular, it focuses on programming 
computers to process and analyze large amounts of natural language data. The result is a computer capable of 
"understanding" the contents of documents, including the contextual nuances of the language within them. 
The technology can then accurately extract information and insights contained in the documents as well as 
categorize and organize the documents themselves. Challenges in natural language processing frequently involve 
speech recognition, natural language understanding, and natural language generation.
"""

summary = extractive_summarization(long_text, num_sentences=2)
print(summary)
```

This cheat sheet covers the essential aspects of NLTK for natural language processing tasks. The library is particularly strong in academic and research contexts, providing comprehensive tools for text analysis, linguistic processing, and building NLP applications. Its extensive corpus collection and built-in algorithms make it an excellent choice for learning NLP concepts and rapid prototyping.