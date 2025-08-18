# Transformers (Hugging Face) Cheat Sheet

Transformers is Hugging Face's flagship library providing state-of-the-art machine learning models for PyTorch, TensorFlow, and JAX. It offers thousands of pretrained models to perform tasks on different domains like text, vision, and audio.

## Installation

```bash
# Basic installation
pip install transformers

# With PyTorch
pip install transformers[torch]

# With TensorFlow
pip install transformers[tf]

# With additional dependencies
pip install transformers[torch,vision,audio]

# Development version
pip install git+https://github.com/huggingface/transformers.git

# With specific backend
pip install transformers torch torchvision torchaudio
```

## Basic Setup

```python
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoModelForMaskedLM,
    pipeline, Trainer, TrainingArguments,
    BertTokenizer, BertModel, GPT2LMHeadModel,
    T5ForConditionalGeneration, T5Tokenizer
)

import torch
import numpy as np
from datasets import load_dataset
```

## Core Functionality

### Pipeline API (Quickstart)

```python
# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=50, num_return_sequences=2)
print(result[0]['generated_text'])

# Question answering
qa_pipeline = pipeline("question-answering")
context = "The capital of France is Paris. Paris is known for the Eiffel Tower."
question = "What is the capital of France?"
answer = qa_pipeline(question=question, context=context)
print(answer)  # {'answer': 'Paris', 'score': 0.9999, 'start': 23, 'end': 28}

# Named Entity Recognition
ner = pipeline("ner", aggregation_strategy="simple")
text = "My name is Wolfgang and I live in Berlin"
entities = ner(text)
print(entities)

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
article = "Your long article text here..."
summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])

# Translation
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
print(result[0]['translation_text'])  # Bonjour, comment allez-vous?

# Fill mask
unmasker = pipeline("fill-mask")
result = unmasker("Paris is the capital of <mask>.")
print(result[0]['token_str'])  # France
```

### Model and Tokenizer Loading

```python
# Auto classes (recommended)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Specific model classes
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Loading with specific configurations
from transformers import BertConfig
config = BertConfig.from_pretrained("bert-base-uncased")
config.num_hidden_layers = 6  # Modify configuration
model = BertModel.from_pretrained("bert-base-uncased", config=config)

# Loading from local files
tokenizer = AutoTokenizer.from_pretrained("./my_model_directory")
model = AutoModel.from_pretrained("./my_model_directory")

# Loading with specific torch dtype
model = AutoModel.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Tokenization

```python
# Basic tokenization
text = "Hello, how are you today?"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['hello', ',', 'how', 'are', 'you', 'today', '?']

# Encoding (text to tokens to IDs)
encoding = tokenizer(text)
print(encoding['input_ids'])  # [101, 7592, 1010, 2129, 2024, 2017, 2651, 1029, 102]
print(encoding['attention_mask'])  # [1, 1, 1, 1, 1, 1, 1, 1, 1]

# Batch encoding
texts = ["Hello world!", "How are you?", "Fine, thanks!"]
batch_encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"  # PyTorch tensors
)

# Decoding (IDs back to text)
decoded = tokenizer.decode(encoding['input_ids'])
print(decoded)  # [CLS] hello, how are you today? [SEP]

# Advanced tokenization options
encoding = tokenizer(
    text,
    add_special_tokens=True,    # Add [CLS] and [SEP]
    max_length=512,            # Maximum sequence length
    padding="max_length",      # Pad to max_length
    truncation=True,           # Truncate if longer
    return_attention_mask=True, # Return attention masks
    return_token_type_ids=True, # Return token type IDs (for BERT)
    return_tensors="pt"        # Return PyTorch tensors
)
```

## Common Use Cases

### Text Classification

```python
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

# Load model for classification
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_sentiment(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = F.softmax(outputs.logits, dim=-1)
    
    # Get predicted class
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    # Map to labels (model specific)
    labels = ["negative", "neutral", "positive"]
    return {
        "label": labels[predicted_class],
        "confidence": confidence
    }

# Usage
result = classify_sentiment("I love this new feature!")
print(result)  # {'label': 'positive', 'confidence': 0.9234}
```

### Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add pad token
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.7, num_return_sequences=1):
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    # Decode results
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

# Usage
texts = generate_text("The future of artificial intelligence", max_length=80)
for text in texts:
    print(text)
```

### Question Answering

```python
from transformers import AutoModelForQuestionAnswering

# Load QA model
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(context, question):
    # Tokenize
    inputs = tokenizer.encode_plus(
        question, context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
    
    # Find the best answer
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1
    
    # Decode answer
    input_ids = inputs['input_ids'][0]
    answer_tokens = input_ids[start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Calculate confidence
    confidence = (start_scores[0][start_idx] + end_scores[0][end_idx-1]).item()
    
    return {
        "answer": answer,
        "confidence": confidence,
        "start": start_idx.item(),
        "end": end_idx.item()
    }

# Usage
context = """
The Transformer architecture was introduced in the paper "Attention Is All You Need" 
by Vaswani et al. in 2017. It revolutionized natural language processing by using 
self-attention mechanisms instead of recurrent layers.
"""

question = "When was the Transformer architecture introduced?"
result = answer_question(context, question)
print(result)  # {'answer': '2017', 'confidence': 15.23, ...}
```

### Text Summarization

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load T5 model for summarization
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=150, min_length=50):
    # T5 requires task prefix
    input_text = "summarize: " + text
    
    # Tokenize
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Usage
article = """
Artificial Intelligence (AI) has become increasingly important in modern technology. 
Machine learning algorithms can now process vast amounts of data and make predictions 
with remarkable accuracy. Deep learning, a subset of machine learning, uses neural 
networks with multiple layers to learn complex patterns in data. This technology 
powers many applications we use daily, from search engines to recommendation systems.
"""

summary = summarize_text(article)
print(summary)
```

## Advanced Features

### Fine-tuning Models

```python
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import torch

# Prepare dataset
def prepare_dataset():
    # Your data preparation logic
    texts = ["positive example", "negative example", ...]
    labels = [1, 0, ...]  # Binary classification
    
    return Dataset.from_dict({
        "text": texts,
        "labels": labels
    })

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128
    )

# Load model for training
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Prepare data
train_dataset = prepare_dataset()
train_dataset = train_dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Define compute metrics function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Use validation set in practice
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_model")
```

### Custom Model Architecture

```python
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class CustomBertClassifier(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
        
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

# Usage
config = AutoConfig.from_pretrained("bert-base-uncased")
model = CustomBertClassifier(config, num_labels=3)
```

### Working with Different Modalities

```python
# Vision-Language Models (CLIP)
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Process inputs
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True
)

# Get similarities
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)  # Probability for each text description

# Audio Models (Wav2Vec2)
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

# Load audio model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Process audio file
audio, sampling_rate = librosa.load("path_to_audio.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# Get transcription
with torch.no_grad():
    logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print(transcription)
```

## Integration with Other Libraries

### With Datasets Library

```python
from datasets import load_dataset, Dataset, DatasetDict

# Load popular datasets
dataset = load_dataset("imdb")  # Movie reviews
dataset = load_dataset("squad")  # Question answering
dataset = load_dataset("glue", "sst2")  # Sentiment analysis

# Create custom dataset
data = {
    "text": ["Great movie!", "Terrible film.", "Average story."],
    "label": [2, 0, 1]  # positive, negative, neutral
}
custom_dataset = Dataset.from_dict(data)

# Tokenize dataset
def tokenize_data(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = custom_dataset.map(tokenize_data, batched=True)

# Split dataset
train_test = custom_dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': train_test['train'],
    'test': train_test['test']
})
```

### With PyTorch Lightning

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class TransformersLightningModule(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", num_labels=2, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

# Usage
model = TransformersLightningModule()
trainer = pl.Trainer(max_epochs=3, gpus=1 if torch.cuda.is_available() else 0)
trainer.fit(model, train_dataloader, val_dataloader)
```

### With Gradio for Web Apps

```python
import gradio as gr
from transformers import pipeline

# Create sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", 
                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"Sentiment: {result['label']} (Confidence: {result['score']:.3f})"

# Create Gradio interface
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter text to analyze..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Analyze the sentiment of your text using RoBERTa model"
)

# Launch the app
interface.launch()
```

## Best Practices

### Memory and Performance Optimization

```python
# 1. Use appropriate model sizes
model = AutoModel.from_pretrained(
    "distilbert-base-uncased",  # Smaller, faster alternative to BERT
    torch_dtype=torch.float16,  # Half precision for memory efficiency
    device_map="auto"           # Automatic device placement
)

# 2. Batch processing for efficiency
def batch_inference(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_encoding = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**batch_encoding)
            results.extend(outputs.logits.cpu().numpy())
    
    return results

# 3. Gradient checkpointing for training large models
model.gradient_checkpointing_enable()

# 4. Use DataLoader with multiple workers
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True
)
```

### Model Evaluation and Monitoring

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, tokenizer, test_texts, test_labels):
    predictions = []
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)
    
    # Classification report
    report = classification_report(test_labels, predictions, output_dict=True)
    print(classification_report(test_labels, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return report, cm

# Usage
report, cm = evaluate_model(model, tokenizer, test_texts, test_labels)
```

### Error Handling and Robust Inference

```python
def robust_inference(text, model, tokenizer, max_retries=3):
    """Robust inference with error handling"""
    for attempt in range(max_retries):
        try:
            # Validate input
            if not isinstance(text, str) or not text.strip():
                return {"error": "Invalid input text"}
            
            # Tokenize with length check
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Check if input is too long even after truncation
            if inputs['input_ids'].shape[1] > 512:
                return {"error": "Text too long even after truncation"}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "success": True
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Inference failed after {max_retries} attempts: {str(e)}"}
            time.sleep(0.1)  # Brief pause before retry
    
    return {"error": "Maximum retries exceeded"}
```

## Real-world Examples

### Complete Text Classification System

```python
class TextClassificationSystem:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        import re
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = re.sub(r'#\w+', '', text)     # Remove hashtags
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.strip()
    
    def predict(self, text):
        """Make prediction on single text"""
        cleaned_text = self.preprocess_text(text)
        
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "prediction": self.label_map[predicted_class],
            "confidence": confidence,
            "all_scores": {
                self.label_map[i]: predictions[0][i].item() 
                for i in range(len(self.label_map))
            }
        }
    
    def predict_batch(self, texts, batch_size=32):
        """Make predictions on batch of texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            cleaned_texts = [self.preprocess_text(text) for text in batch_texts]
            
            inputs = self.tokenizer(
                cleaned_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(predictions, dim=-1)
                
                for j, (text, cleaned_text) in enumerate(zip(batch_texts, cleaned_texts)):
                    pred_class = predicted_classes[j].item()
                    confidence = predictions[j][pred_class].item()
                    
                    results.append({
                        "text": text,
                        "cleaned_text": cleaned_text,
                        "prediction": self.label_map[pred_class],
                        "confidence": confidence
                    })
        
        return results

# Usage
classifier = TextClassificationSystem()

# Single prediction
result = classifier.predict("I absolutely love this new product! It's amazing!")
print(result)

# Batch prediction
texts = [
    "Great service and friendly staff!",
    "Terrible experience, would not recommend.",
    "It was okay, nothing special."
]
results = classifier.predict_batch(texts)
for result in results:
    print(f"{result['prediction']}: {result['text']} (confidence: {result['confidence']:.3f})")
```

### Multi-task NLP Pipeline

```python
class MultiTaskNLPPipeline:
    def __init__(self):
        # Initialize different models for different tasks
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.ner_model = pipeline("ner", aggregation_strategy="simple")
        self.qa_model = pipeline("question-answering")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def analyze_text(self, text, tasks=["sentiment", "ner", "summary"]):
        """Comprehensive text analysis"""
        results = {"original_text": text}
        
        if "sentiment" in tasks:
            sentiment = self.sentiment_analyzer(text)[0]
            results["sentiment"] = {
                "label": sentiment["label"],
                "confidence": sentiment["score"]
            }
        
        if "ner" in tasks:
            entities = self.ner_model(text)
            results["entities"] = [
                {
                    "text": entity["word"],
                    "label": entity["entity_group"],
                    "confidence": entity["score"]
                }
                for entity in entities
            ]
        
        if "summary" in tasks and len(text.split()) > 30:
            try:
                summary = self.summarizer(text, max_length=100, min_length=30)[0]
                results["summary"] = summary["summary_text"]
            except:
                results["summary"] = "Text too short for summarization"
        
        return results
    
    def answer_questions(self, context, questions):
        """Answer multiple questions about a context"""
        answers = []
        for question in questions:
            try:
                answer = self.qa_model(question=question, context=context)
                answers.append({
                    "question": question,
                    "answer": answer["answer"],
                    "confidence": answer["score"]
                })
            except:
                answers.append({
                    "question": question,
                    "answer": "Could not answer",
                    "confidence": 0.0
                })
        return answers

# Usage
nlp_pipeline = MultiTaskNLPPipeline()

text = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
Apple is the world's largest technology company by revenue and the world's most valuable company. 
The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. 
Apple's products include the iPhone, iPad, Mac, Apple Watch, and Apple TV.
"""

# Comprehensive analysis
analysis = nlp_pipeline.analyze_text(text)
print("Analysis Results:")
print(f"Sentiment: {analysis['sentiment']['label']} ({analysis['sentiment']['confidence']:.3f})")
print(f"Entities: {[e['text'] for e in analysis['entities']]}")
print(f"Summary: {analysis['summary']}")

# Question answering
questions = [
    "Who founded Apple?",
    "Where is Apple headquartered?",
    "What products does Apple make?"
]
answers = nlp_pipeline.answer_questions(text, questions)
for qa in answers:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']} (confidence: {qa['confidence']:.3f})")
```

This comprehensive cheat sheet covers the essential aspects of the Transformers library. The library's strength lies in its unified API across different models and tasks, extensive model zoo, and seamless integration with the broader ML ecosystem. It's the go-to library for state-of-the-art NLP applications and research.