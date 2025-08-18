# Sentence-Transformers (UKPLab) Cheat Sheet

Sentence-Transformers is a Python framework for state-of-the-art sentence, text, and image embeddings. It provides an easy way to compute dense vector representations for sentences, paragraphs, and images, enabling semantic similarity computation, clustering, and semantic search.

## Installation

```bash
# Basic installation
pip install sentence-transformers

# With optional dependencies
pip install sentence-transformers[train]

# Development version
pip install git+https://github.com/UKPLab/sentence-transformers.git

# With specific backends
pip install sentence-transformers torch torchvision
pip install sentence-transformers tensorflow
```

## Basic Setup

```python
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
```

## Core Functionality

### Loading Models

```python
# Popular pre-trained models
model = SentenceTransformer('all-MiniLM-L6-v2')  # Good balance of quality vs speed
model = SentenceTransformer('all-mpnet-base-v2')  # High quality
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Paraphrase detection
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # Question answering

# Multilingual models
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Specialized models
model = SentenceTransformer('msmarco-distilbert-base-v4')  # Web search
model = SentenceTransformer('nli-distilroberta-base-v2')  # Natural language inference

# Load from local path
model = SentenceTransformer('/path/to/model')

# Load with specific device
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

### Basic Encoding

```python
# Single sentence encoding
sentence = "This is a sample sentence."
embedding = model.encode(sentence)
print(f"Embedding shape: {embedding.shape}")  # (384,) for MiniLM models

# Multiple sentences
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]

embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 384)

# Batch processing with progress bar
embeddings = model.encode(sentences, show_progress_bar=True)

# Convert to tensor
embeddings = model.encode(sentences, convert_to_tensor=True)
print(type(embeddings))  # torch.Tensor

# Normalize embeddings (for cosine similarity)
embeddings = model.encode(sentences, normalize_embeddings=True)
```

### Similarity Computation

```python
# Compute similarity matrix
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]

embeddings = model.encode(sentences)

# Cosine similarity
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

# Pairwise similarity
similarities = util.pytorch_cos_sim(embeddings, embeddings)

# Find most similar pairs
pairs = []
for i in range(len(similarities)):
    for j in range(i+1, len(similarities)):
        pairs.append((i, j, similarities[i][j].item()))

# Sort by similarity
pairs.sort(key=lambda x: x[2], reverse=True)
print(f"Most similar pair: sentences {pairs[0][0]} and {pairs[0][1]} with score {pairs[0][2]:.4f}")
```

## Common Use Cases

### Semantic Search

```python
import numpy as np

# Create a corpus of documents
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey."
]

# Encode corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Query
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field."
]

# Find the closest 5 sentences to each query
top_k = min(5, len(corpus))

for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine-similarities between query and corpus
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # Sort results
    top_results = torch.topk(cos_scores, k=top_k)

    print(f"\nQuery: {query}")
    print(f"Top {top_k} most similar sentences in corpus:")
    
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"(Score: {score:.4f}) {corpus[idx]}")
```

### Advanced Semantic Search with Faiss

```python
import faiss
import numpy as np

def create_faiss_index(embeddings):
    """Create FAISS index for fast similarity search"""
    dimension = embeddings.shape[1]
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
    
    # Normalize embeddings for cosine similarity
    embeddings_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
    faiss.normalize_L2(embeddings_np)
    
    # Add embeddings to index
    index.add(embeddings_np.astype('float32'))
    
    return index

def semantic_search_faiss(query, model, index, corpus, top_k=5):
    """Perform semantic search using FAISS"""
    # Encode query
    query_embedding = model.encode([query])
    
    # Normalize
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'score': float(score),
            'text': corpus[idx],
            'index': int(idx)
        })
    
    return results

# Usage
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
index = create_faiss_index(corpus_embeddings)

query = "A person is eating food"
results = semantic_search_faiss(query, model, index, corpus)

print(f"Query: {query}")
for result in results:
    print(f"Score: {result['score']:.4f} - {result['text']}")
```

### Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample sentences for clustering
sentences = [
    # Sports
    "The football game was exciting.",
    "Basketball is my favorite sport.",
    "Tennis requires good coordination.",
    
    # Food
    "This pizza tastes amazing.",
    "I love cooking Italian food.",
    "The restaurant serves great pasta.",
    
    # Technology
    "Machine learning is fascinating.",
    "AI will change the world.",
    "Programming languages are evolving.",
    
    # Weather
    "It's a beautiful sunny day.",
    "The weather is getting cold.",
    "Rain is expected tomorrow."
]

# Get embeddings
embeddings = model.encode(sentences)

# Perform clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(embeddings)

# Group sentences by cluster
clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignments):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []
    clustered_sentences[cluster_id].append(sentences[sentence_id])

# Print results
for cluster_id, cluster_sentences in clustered_sentences.items():
    print(f"Cluster {cluster_id + 1}:")
    for sentence in cluster_sentences:
        print(f"  - {sentence}")
    print()
```

### Paraphrase Detection

```python
def find_paraphrases(sentences, threshold=0.7):
    """Find paraphrases in a list of sentences"""
    embeddings = model.encode(sentences)
    similarity_matrix = util.cos_sim(embeddings, embeddings)
    
    paraphrases = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarity = similarity_matrix[i][j].item()
            if similarity > threshold:
                paraphrases.append({
                    'sentence1': sentences[i],
                    'sentence2': sentences[j],
                    'similarity': similarity
                })
    
    return sorted(paraphrases, key=lambda x: x['similarity'], reverse=True)

# Example sentences with some paraphrases
test_sentences = [
    "The cat is sleeping on the couch.",
    "A feline is resting on the sofa.",
    "The dog is barking loudly.",
    "The weather is nice today.",
    "It's a beautiful day outside.",
    "The car is red.",
    "The canine is making loud sounds."
]

paraphrases = find_paraphrases(test_sentences, threshold=0.6)

print("Potential paraphrases found:")
for para in paraphrases:
    print(f"Similarity: {para['similarity']:.3f}")
    print(f"1: {para['sentence1']}")
    print(f"2: {para['sentence2']}")
    print("-" * 50)
```

### Question Answering with Retrieval

```python
def qa_retrieval_system(questions, contexts, model, top_k=3):
    """Simple QA retrieval system using sentence embeddings"""
    
    # Encode all contexts
    context_embeddings = model.encode(contexts, convert_to_tensor=True)
    
    results = []
    for question in questions:
        # Encode question
        question_embedding = model.encode(question, convert_to_tensor=True)
        
        # Find most similar contexts
        similarities = util.cos_sim(question_embedding, context_embeddings)[0]
        top_indices = torch.topk(similarities, k=min(top_k, len(contexts)))[1]
        
        # Get top contexts
        top_contexts = [contexts[idx] for idx in top_indices]
        top_scores = [similarities[idx].item() for idx in top_indices]
        
        results.append({
            'question': question,
            'top_contexts': list(zip(top_contexts, top_scores))
        })
    
    return results

# Example usage
contexts = [
    "Paris is the capital of France and its largest city.",
    "London is the capital of England and the United Kingdom.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy and home to Vatican City.",
    "Tokyo is the capital of Japan and the world's most populous metropolitan area."
]

questions = [
    "What is the capital of France?",
    "Which city is the capital of Germany?",
    "Tell me about the capital of Italy."
]

qa_results = qa_retrieval_system(questions, contexts, model)

for result in qa_results:
    print(f"Question: {result['question']}")
    print("Most relevant contexts:")
    for context, score in result['top_contexts']:
        print(f"  Score: {score:.3f} - {context}")
    print()
```

## Advanced Features

### Custom Model Training

```python
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data
train_examples = [
    InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    InputExample(texts=['Another pair', 'Completely different'], label=0.3)
]

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path='./my-sentence-transformer'
)

# For triplet training (anchor, positive, negative)
train_examples_triplet = [
    InputExample(texts=['Anchor sentence', 'Positive sentence', 'Negative sentence'])
]

train_dataloader_triplet = DataLoader(train_examples_triplet, shuffle=True, batch_size=16)
train_loss_triplet = losses.TripletLoss(model)

model.fit(
    train_objectives=[(train_dataloader_triplet, train_loss_triplet)],
    epochs=1,
    output_path='./my-triplet-model'
)
```

### Cross-Encoders for Reranking

```python
from sentence_transformers.cross_encoder import CrossEncoder

# Load cross-encoder model
cross_encoder = CrossEncoder('ms-marco-MiniLM-L-6-v2')

def rerank_results(query, candidates, cross_encoder, top_k=5):
    """Rerank search results using a cross-encoder"""
    
    # Create query-candidate pairs
    pairs = [[query, candidate] for candidate in candidates]
    
    # Get cross-encoder scores
    scores = cross_encoder.predict(pairs)
    
    # Sort by score
    ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    return ranked_results[:top_k]

# Example: Two-stage retrieval and reranking
def two_stage_search(query, corpus, bi_encoder, cross_encoder, 
                     retrieve_top_k=20, rerank_top_k=5):
    """Two-stage search: bi-encoder retrieval + cross-encoder reranking"""
    
    # Stage 1: Fast retrieval with bi-encoder
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(retrieve_top_k, len(corpus)))
    
    # Get candidate texts
    candidates = [corpus[idx] for idx in top_results[1]]
    
    # Stage 2: Reranking with cross-encoder
    reranked = rerank_results(query, candidates, cross_encoder, rerank_top_k)
    
    return reranked

# Usage
query = "Information about machine learning"
corpus = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to interpret visual information.",
    "Reinforcement learning trains agents through trial and error."
]

bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('ms-marco-MiniLM-L-6-v2')

results = two_stage_search(query, corpus, bi_encoder, cross_encoder)

print(f"Query: {query}")
print("Reranked results:")
for i, (text, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f} - {text}")
```

### Multi-lingual Embeddings

```python
# Load multilingual model
multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Sentences in different languages
sentences = [
    "Hello, how are you?",           # English
    "Hola, ¿cómo estás?",           # Spanish
    "Bonjour, comment allez-vous?",  # French
    "Hallo, wie geht es dir?",      # German
    "Ciao, come stai?",             # Italian
    "こんにちは、元気ですか？",           # Japanese
    "你好，你好吗？"                   # Chinese
]

# Get embeddings
embeddings = multilingual_model.encode(sentences)

# Compute similarity matrix
similarity_matrix = util.cos_sim(embeddings, embeddings)

print("Cross-lingual similarity matrix:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i != j:
            sim = similarity_matrix[i][j].item()
            if sim > 0.5:  # Only show high similarities
                print(f"Similarity: {sim:.3f}")
                print(f"  '{sent1}' <-> '{sent2}'")
```

### Working with Images (CLIP)

```python
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO

# Load CLIP model
clip_model = SentenceTransformer('clip-ViT-B-32')

# Load images
image_urls = [
    "https://example.com/cat.jpg",
    "https://example.com/dog.jpg"
]

images = []
for url in image_urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    images.append(img)

# Text descriptions
texts = [
    "A photo of a cat",
    "A photo of a dog",
    "A picture of a bird",
    "An image of a car"
]

# Get embeddings
image_embeddings = clip_model.encode(images)
text_embeddings = clip_model.encode(texts)

# Compute similarity between images and texts
similarities = util.cos_sim(image_embeddings, text_embeddings)

print("Image-Text Similarities:")
for i, img_url in enumerate(image_urls):
    print(f"\nImage {i+1}: {img_url}")
    for j, text in enumerate(texts):
        sim = similarities[i][j].item()
        print(f"  '{text}': {sim:.3f}")
```

## Integration with Other Libraries

### With Pandas for Data Analysis

```python
import pandas as pd

# Create sample dataset
data = {
    'text': [
        "I love this product! It's amazing!",
        "Great quality and fast shipping.",
        "Terrible experience, would not recommend.",
        "Average product, nothing special.",
        "Outstanding customer service!"
    ],
    'rating': [5, 4, 1, 3, 5]
}

df = pd.DataFrame(data)

# Add embeddings
embeddings = model.encode(df['text'].tolist())
df['embedding'] = embeddings.tolist()

# Find similar reviews
def find_similar_reviews(query_text, df, model, top_k=3):
    query_embedding = model.encode([query_text])
    
    # Compute similarities
    similarities = []
    for i, row in df.iterrows():
        sim = cosine_similarity([query_embedding[0]], [row['embedding']])[0][0]
        similarities.append(sim)
    
    df['similarity'] = similarities
    return df.nlargest(top_k, 'similarity')[['text', 'rating', 'similarity']]

# Usage
query = "excellent customer support"
similar_reviews = find_similar_reviews(query, df, model)
print(similar_reviews)
```

### With Streamlit for Web Apps

```python
import streamlit as st
import plotly.express as px
from sklearn.manifold import TSNE

st.title("Sentence Similarity Explorer")

# Text input
user_texts = st.text_area("Enter sentences (one per line):", 
                         value="The weather is nice today.\nIt's a beautiful day.\nThe car is red.")

if user_texts:
    sentences = [s.strip() for s in user_texts.split('\n') if s.strip()]
    
    # Compute embeddings
    embeddings = model.encode(sentences)
    
    # Compute similarity matrix
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    
    # Display similarity matrix
    st.subheader("Similarity Matrix")
    fig = px.imshow(similarity_matrix, 
                    labels=dict(x="Sentences", y="Sentences"),
                    text_auto=True)
    st.plotly_chart(fig)
    
    # 2D visualization using t-SNE
    if len(sentences) > 2:
        st.subheader("2D Embedding Visualization")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        fig_scatter = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
                               hover_data=[sentences],
                               title="Sentence Embeddings (t-SNE)")
        st.plotly_chart(fig_scatter)
```

## Best Practices

### Performance Optimization

```python
# 1. Use appropriate model sizes
models_by_performance = {
    'fastest': 'all-MiniLM-L6-v2',
    'balanced': 'all-mpnet-base-v2', 
    'highest_quality': 'sentence-transformers/gtr-t5-large'
}

# 2. Batch processing for large datasets
def encode_large_dataset(texts, model, batch_size=32):
    """Efficiently encode large datasets"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# 3. Use appropriate precision
embeddings = model.encode(sentences, precision='float32')  # vs 'float64'

# 4. Normalize embeddings if using cosine similarity
embeddings = model.encode(sentences, normalize_embeddings=True)

# 5. Use GPU when available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
```

### Memory Management

```python
import gc
from typing import Iterator, List

def process_large_corpus(corpus: List[str], 
                        model: SentenceTransformer,
                        batch_size: int = 1000) -> Iterator[np.ndarray]:
    """Process large corpus in chunks to manage memory"""
    
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        
        # Encode batch
        embeddings = model.encode(batch, convert_to_numpy=True)
        
        yield embeddings
        
        # Clean up
        gc.collect()

# Usage for very large datasets
def save_embeddings_chunked(corpus, model, output_path, batch_size=1000):
    """Save embeddings for large corpus in chunks"""
    all_embeddings = []
    
    for chunk_embeddings in process_large_corpus(corpus, model, batch_size):
        all_embeddings.append(chunk_embeddings)
    
    # Concatenate all embeddings
    final_embeddings = np.vstack(all_embeddings)
    np.save(output_path, final_embeddings)
    
    return final_embeddings
```

### Model Selection Guidelines

```python
def recommend_model(use_case, performance_priority='balanced'):
    """Recommend model based on use case and performance requirements"""
    
    recommendations = {
        'semantic_search': {
            'fast': 'all-MiniLM-L6-v2',
            'balanced': 'all-mpnet-base-v2',
            'quality': 'sentence-transformers/gtr-t5-large'
        },
        'question_answering': {
            'fast': 'multi-qa-MiniLM-L6-cos-v1',
            'balanced': 'multi-qa-mpnet-base-cos-v1',
            'quality': 'sentence-transformers/gtr-t5-xl'
        },
        'paraphrase_detection': {
            'fast': 'paraphrase-MiniLM-L6-v2',
            'balanced': 'paraphrase-mpnet-base-v2',
            'quality': 'sentence-transformers/gtr-t5-large'
        },
        'multilingual': {
            'fast': 'paraphrase-multilingual-MiniLM-L12-v2',
            'balanced': 'sentence-transformers/LaBSE',
            'quality': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        }
    }
    
    return recommendations.get(use_case, {}).get(performance_priority, 'all-MiniLM-L6-v2')

# Usage
model_name = recommend_model('semantic_search', 'fast')
print(f"Recommended model: {model_name}")
```

## Real-world Examples

### Complete Semantic Search Engine

```python
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

class SemanticSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the search engine"""
        self.documents.extend(documents)
        
        # Extract text for embedding
        texts = [doc.get('text', str(doc)) for doc in documents]
        
        # Compute embeddings
        new_embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarities
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            doc = self.documents[idx.item()].copy()
            doc['similarity_score'] = score.item()
            results.append(doc)
        
        return results
    
    def save(self, path: str):
        """Save the search engine state"""
        state = {
            'documents': self.documents,
            'embeddings': self.embeddings.cpu().numpy() if self.embeddings is not None else None,
            'model_name': self.model._modules['0'].auto_model.config._name_or_path
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load the search engine state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.documents = state['documents']
        if state['embeddings'] is not None:
            self.embeddings = torch.tensor(state['embeddings'])

# Usage example
search_engine = SemanticSearchEngine()

# Add documents
documents = [
    {"title": "Machine Learning Basics", "text": "Introduction to machine learning algorithms and concepts", "category": "AI"},
    {"title": "Python Programming", "text": "Learn Python programming from scratch", "category": "Programming"},
    {"title": "Data Science Guide", "text": "Comprehensive guide to data science and analytics", "category": "Data"},
    {"title": "Neural Networks", "text": "Deep learning and neural network architectures", "category": "AI"},
]

search_engine.add_documents(documents)

# Search
results = search_engine.search("artificial intelligence and machine learning", top_k=3)

print("Search Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']} (Score: {result['similarity_score']:.3f})")
    print(f"   Category: {result['category']}")
    print(f"   Text: {result['text']}")
    print()

# Save for later use
search_engine.save("my_search_engine.pkl")
```

### Document Clustering and Topic Analysis

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class DocumentAnalyzer:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
    
    def analyze_documents(self, documents, num_clusters=5):
        """Analyze documents: embeddings, clustering, and visualization"""
        
        # Get embeddings
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create results
        results = {
            'documents': documents,
            'embeddings': embeddings,
            'cluster_labels': cluster_labels,
            'embeddings_2d': embeddings_2d,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        return results
    
    def visualize_clusters(self, results):
        """Visualize document clusters"""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(results['embeddings_2d'][:, 0], 
                            results['embeddings_2d'][:, 1], 
                            c=results['cluster_labels'], 
                            cmap='tab10', 
                            alpha=0.7)
        
        plt.colorbar(scatter)
        plt.title('Document Clusters (2D PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Add some document texts as annotations
        for i in range(0, len(results['documents']), max(1, len(results['documents'])//10)):
            plt.annotate(results['documents'][i][:30] + '...', 
                        (results['embeddings_2d'][i, 0], results['embeddings_2d'][i, 1]),
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_summaries(self, results, max_examples=3):
        """Get representative examples for each cluster"""
        summaries = {}
        
        for cluster_id in range(len(set(results['cluster_labels']))):
            # Get documents in this cluster
            cluster_docs = [doc for i, doc in enumerate(results['documents']) 
                          if results['cluster_labels'][i] == cluster_id]
            
            # Get embeddings for this cluster
            cluster_embeddings = results['embeddings'][results['cluster_labels'] == cluster_id]
            cluster_center = results['cluster_centers'][cluster_id]
            
            # Find most representative documents (closest to center)
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest_indices = np.argsort(distances)[:max_examples]
            
            representative_docs = [cluster_docs[idx] for idx in closest_indices]
            
            summaries[f'Cluster {cluster_id}'] = {
                'size': len(cluster_docs),
                'representative_docs': representative_docs
            }
        
        return summaries

# Example usage
analyzer = DocumentAnalyzer()

# Sample documents (in practice, load from your data source)
documents = [
    "Machine learning algorithms for data analysis",
    "Python programming tutorial for beginners",
    "Natural language processing with transformers",
    "Web development using React and Node.js",
    "Deep learning neural networks architecture",
    "Database design and SQL optimization",
    "Computer vision and image recognition",
    "JavaScript frameworks comparison",
    "Data science workflow and best practices",
    "Mobile app development with Flutter"
]

# Analyze documents
results = analyzer.analyze_documents(documents, num_clusters=3)

# Visualize
analyzer.visualize_clusters(results)

# Get cluster summaries
summaries = analyzer.get_cluster_summaries(results)
print("Cluster Analysis:")
for cluster_name, info in summaries.items():
    print(f"\n{cluster_name} ({info['size']} documents):")
    for doc in info['representative_docs']:
        print(f"  - {doc}")
```

This comprehensive cheat sheet covers the essential aspects of Sentence-Transformers. The library excels at creating meaningful vector representations of text, enabling powerful semantic search, similarity computation, and clustering applications. Its ease of use and extensive model collection make it ideal for both research and production use cases.