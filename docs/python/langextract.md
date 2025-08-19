# LangExtract Cheat Sheet

LangExtract is Google's open-source Python library for extracting structured information from unstructured text using Large Language Models (LLMs). It provides precise source grounding, interactive visualizations, and supports multiple model providers.

## Installation

```bash
# Basic installation
pip install langextract

# For development (from source)
git clone https://github.com/google/langextract.git
cd langextract
pip install -e .
```

## Quick Start

```python
import langextract as lx

# Basic extraction
result = lx.extract(
    text_or_documents="Your unstructured text here...",
    prompt_description="Extract names, dates, and locations",
    examples=[
        {"input": "John visited Paris on May 15th", 
         "output": {"names": ["John"], "places": ["Paris"], "dates": ["May 15th"]}}
    ],
    model_id="gemini-2.0-flash-exp"
)

# Access results
print(result.extractions)
print(result.visualize())  # Interactive HTML visualization
```

## Core Components

### 1. Basic Extraction

```python
import langextract as lx

# Simple extraction with few-shot examples
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract character names and their emotions",
    examples=[
        {
            "input": "Alice felt happy about the good news",
            "output": {
                "characters": [{"name": "Alice", "emotion": "happy"}]
            }
        }
    ],
    model_id="gemini-2.0-flash-exp"
)

# Check extraction results
for extraction in result.extractions:
    print(f"Text: {extraction.text}")
    print(f"Data: {extraction.data}")
    print(f"Source spans: {extraction.source_spans}")
```

### 2. Document Processing

```python
# Process multiple documents
documents = [
    {"text": "Document 1 content...", "metadata": {"source": "doc1.txt"}},
    {"text": "Document 2 content...", "metadata": {"source": "doc2.txt"}},
]

result = lx.extract(
    text_or_documents=documents,
    prompt_description="Extract key findings and recommendations",
    examples=[...],
    model_id="gemini-2.0-flash-exp"
)
```

### 3. Model Configuration

```python
# Using different models
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.0-flash-exp",  # Recommended for speed
    # model_id="gemini-2.0-pro",      # For complex reasoning
    # model_id="gpt-4o-mini",         # OpenAI alternative
)

# Configure model parameters
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.0-flash-exp",
    generation_config={
        "temperature": 0.1,
        "max_output_tokens": 8192,
        "top_p": 0.95
    }
)
```

## Advanced Features

### 1. Source Grounding & Visualization

```python
# Extract with precise source tracking
result = lx.extract(
    text_or_documents=long_text,
    prompt_description="Extract medical conditions and treatments",
    examples=[...],
    model_id="gemini-2.0-flash-exp"
)

# Generate interactive visualization
html_viz = result.visualize()

# Save visualization to file
with open("extraction_results.html", "w") as f:
    f.write(html_viz)

# Access source spans for each extraction
for extraction in result.extractions:
    for entity in extraction.data.get("entities", []):
        spans = extraction.source_spans.get(entity["id"], [])
        print(f"Entity: {entity['text']} found at positions: {spans}")
```

### 2. Complex Schema Extraction

```python
# Define complex extraction schema
medical_examples = [
    {
        "input": "Patient John Smith, 45, diagnosed with hypertension. Prescribed lisinopril 10mg daily.",
        "output": {
            "patient": {
                "name": "John Smith",
                "age": 45,
                "conditions": ["hypertension"],
                "medications": [
                    {
                        "name": "lisinopril",
                        "dosage": "10mg",
                        "frequency": "daily"
                    }
                ]
            }
        }
    }
]

result = lx.extract(
    text_or_documents=medical_report,
    prompt_description="Extract patient information, conditions, and medications",
    examples=medical_examples,
    model_id="gemini-2.0-flash-exp"
)
```

### 3. Batch Processing

```python
# Process multiple documents efficiently
large_document_set = [
    {"text": doc1_text, "metadata": {"source": "report1.pdf"}},
    {"text": doc2_text, "metadata": {"source": "report2.pdf"}},
    # ... more documents
]

# Parallel processing for large datasets
result = lx.extract(
    text_or_documents=large_document_set,
    prompt_description="Extract key metrics and insights",
    examples=examples,
    model_id="gemini-2.0-flash-exp",
    max_workers=4  # Control parallel processing
)

# Process results per document
for doc_result in result.extractions:
    source = doc_result.metadata.get("source", "unknown")
    print(f"Results from {source}: {doc_result.data}")
```

### 4. Custom Output Parsers

```python
# Define custom parsing logic
def parse_financial_data(extraction_result):
    """Custom parser for financial documents"""
    parsed_data = {}
    for extraction in extraction_result.extractions:
        # Custom processing logic
        parsed_data[extraction.metadata.get("source")] = {
            "revenue": extraction.data.get("revenue"),
            "expenses": extraction.data.get("expenses"),
            "profit": extraction.data.get("profit")
        }
    return parsed_data

# Use custom parser
result = lx.extract(
    text_or_documents=financial_reports,
    prompt_description="Extract revenue, expenses, and profit figures",
    examples=financial_examples,
    model_id="gemini-2.0-flash-exp"
)

parsed_results = parse_financial_data(result)
```

## Common Use Cases

### 1. Medical Report Processing

```python
medical_examples = [
    {
        "input": "Patient presents with chest pain. ECG shows normal sinus rhythm. Blood pressure 140/90.",
        "output": {
            "symptoms": ["chest pain"],
            "tests": [
                {"name": "ECG", "result": "normal sinus rhythm"},
                {"name": "blood pressure", "result": "140/90"}
            ],
            "assessment": "hypertensive"
        }
    }
]

result = lx.extract(
    text_or_documents=medical_notes,
    prompt_description="Extract symptoms, test results, and clinical assessments",
    examples=medical_examples,
    model_id="gemini-2.0-flash-exp"
)
```

### 2. Legal Document Analysis

```python
legal_examples = [
    {
        "input": "The agreement between ABC Corp and XYZ Inc, dated January 15, 2024, stipulates a payment of $50,000.",
        "output": {
            "parties": ["ABC Corp", "XYZ Inc"],
            "date": "January 15, 2024",
            "financial_terms": [{"amount": "$50,000", "type": "payment"}],
            "document_type": "agreement"
        }
    }
]

result = lx.extract(
    text_or_documents=legal_documents,
    prompt_description="Extract parties, dates, financial terms, and document types",
    examples=legal_examples,
    model_id="gemini-2.0-pro"  # Use Pro for complex legal reasoning
)
```

### 3. Customer Feedback Analysis

```python
feedback_examples = [
    {
        "input": "The product quality is excellent but shipping was slow. Customer service was very helpful.",
        "output": {
            "sentiment": "mixed",
            "aspects": [
                {"category": "product_quality", "sentiment": "positive", "text": "excellent"},
                {"category": "shipping", "sentiment": "negative", "text": "slow"},
                {"category": "customer_service", "sentiment": "positive", "text": "very helpful"}
            ]
        }
    }
]

result = lx.extract(
    text_or_documents=customer_reviews,
    prompt_description="Extract sentiment and specific aspects from customer feedback",
    examples=feedback_examples,
    model_id="gemini-2.0-flash-exp"
)
```

### 4. Research Paper Processing

```python
research_examples = [
    {
        "input": "We conducted a randomized controlled trial with 200 participants. Results showed 85% efficacy (p<0.05).",
        "output": {
            "study_design": "randomized controlled trial",
            "sample_size": 200,
            "key_findings": [
                {"metric": "efficacy", "value": "85%", "significance": "p<0.05"}
            ],
            "study_type": "clinical trial"
        }
    }
]

result = lx.extract(
    text_or_documents=research_papers,
    prompt_description="Extract study methodology, sample sizes, and key findings",
    examples=research_examples,
    model_id="gemini-2.0-flash-exp"
)
```

## Integration Patterns

### 1. With Pandas for Data Analysis

```python
import pandas as pd
import langextract as lx

# Extract structured data
result = lx.extract(
    text_or_documents=documents,
    prompt_description="Extract financial metrics",
    examples=examples,
    model_id="gemini-2.0-flash-exp"
)

# Convert to DataFrame
data_rows = []
for extraction in result.extractions:
    for metric in extraction.data.get("metrics", []):
        data_rows.append({
            "source": extraction.metadata.get("source"),
            "metric_name": metric["name"],
            "value": metric["value"],
            "period": metric.get("period")
        })

df = pd.DataFrame(data_rows)
print(df.groupby("metric_name")["value"].mean())
```

### 2. With LangChain for RAG Systems

```python
import langextract as lx
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Extract structured data first
extraction_result = lx.extract(
    text_or_documents=documents,
    prompt_description="Extract key concepts and definitions",
    examples=examples,
    model_id="gemini-2.0-flash-exp"
)

# Create vector store from extracted data
texts = []
metadatas = []

for extraction in extraction_result.extractions:
    for concept in extraction.data.get("concepts", []):
        texts.append(f"{concept['term']}: {concept['definition']}")
        metadatas.append({
            "source": extraction.metadata.get("source"),
            "term": concept["term"],
            "source_spans": extraction.source_spans.get(concept["id"], [])
        })

vectorstore = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=OpenAIEmbeddings()
)
```

### 3. With FastAPI for API Services

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import langextract as lx

app = FastAPI()

class ExtractionRequest(BaseModel):
    text: str
    task_description: str
    examples: list

class ExtractionResponse(BaseModel):
    extractions: list
    visualization_html: str

@app.post("/extract", response_model=ExtractionResponse)
async def extract_information(request: ExtractionRequest):
    try:
        result = lx.extract(
            text_or_documents=request.text,
            prompt_description=request.task_description,
            examples=request.examples,
            model_id="gemini-2.0-flash-exp"
        )
        
        return ExtractionResponse(
            extractions=[ext.data for ext in result.extractions],
            visualization_html=result.visualize()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Performance Optimization

### 1. Efficient Example Selection

```python
# Use minimal but representative examples
efficient_examples = [
    {
        "input": "Short representative text",
        "output": {"key_field": "value"}
    },
    # Limit to 3-5 high-quality examples
]

# Avoid overly complex output schemas
result = lx.extract(
    text_or_documents=text,
    prompt_description="Clear, specific task description",
    examples=efficient_examples,
    model_id="gemini-2.0-flash-exp"  # Faster for most tasks
)
```

### 2. Chunking Strategy for Long Documents

```python
def chunk_document(text, max_chunk_size=8000):
    """Split document into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), max_chunk_size - 200):  # 200 word overlap
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    
    return chunks

# Process long documents efficiently
long_text = "Very long document content..."
chunks = chunk_document(long_text)

results = []
for chunk in chunks:
    result = lx.extract(
        text_or_documents=chunk,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.0-flash-exp"
    )
    results.append(result)
```

### 3. Caching and Rate Limiting

```python
import time
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_extract(text_hash, prompt, examples_str, model_id):
    """Cache extraction results for identical inputs"""
    return lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=eval(examples_str),  # Be careful with eval in production
        model_id=model_id
    )

def extract_with_rate_limit(text, prompt, examples, model_id, delay=1.0):
    """Add rate limiting between API calls"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    examples_str = str(examples)
    
    result = cached_extract(text_hash, prompt, examples_str, model_id)
    time.sleep(delay)  # Rate limiting
    return result
```

## Error Handling and Debugging

### 1. Robust Error Handling

```python
import langextract as lx
from typing import Optional, List

def safe_extract(
    text: str, 
    prompt: str, 
    examples: List[dict],
    model_id: str = "gemini-2.0-flash-exp",
    max_retries: int = 3
) -> Optional[lx.ExtractionResult]:
    """Extract with error handling and retries"""
    
    for attempt in range(max_retries):
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                model_id=model_id
            )
            return result
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

# Usage
result = safe_extract(text, prompt, examples)
if result:
    print("Extraction successful")
    print(result.extractions)
else:
    print("Extraction failed after all retries")
```

### 2. Validation and Quality Checks

```python
def validate_extraction_result(result: lx.ExtractionResult, expected_fields: List[str]) -> bool:
    """Validate extraction results"""
    if not result or not result.extractions:
        return False
    
    for extraction in result.extractions:
        if not extraction.data:
            return False
        
        # Check for expected fields
        for field in expected_fields:
            if field not in extraction.data:
                print(f"Missing field: {field}")
                return False
    
    return True

# Usage
result = lx.extract(...)
is_valid = validate_extraction_result(result, ["entities", "relationships"])

if not is_valid:
    print("Extraction result validation failed")
```

### 3. Debugging and Inspection

```python
def debug_extraction(result: lx.ExtractionResult):
    """Debug extraction results"""
    print(f"Number of extractions: {len(result.extractions)}")
    
    for i, extraction in enumerate(result.extractions):
        print(f"\nExtraction {i + 1}:")
        print(f"  Text length: {len(extraction.text)}")
        print(f"  Data keys: {list(extraction.data.keys())}")
        print(f"  Source spans: {len(extraction.source_spans)}")
        print(f"  Metadata: {extraction.metadata}")

# Usage
result = lx.extract(...)
debug_extraction(result)
```

## Best Practices

### 1. Example Design

```python
# ✅ Good: Clear, specific examples
good_examples = [
    {
        "input": "Dr. Smith prescribed aspirin 81mg daily for cardiovascular protection",
        "output": {
            "physician": "Dr. Smith",
            "medication": {
                "name": "aspirin",
                "dose": "81mg",
                "frequency": "daily",
                "indication": "cardiovascular protection"
            }
        }
    }
]

# ❌ Avoid: Vague or inconsistent examples
bad_examples = [
    {
        "input": "Some text",
        "output": {"stuff": "things"}
    }
]
```

### 2. Prompt Engineering

```python
# ✅ Good: Specific, actionable prompts
good_prompt = "Extract medication names, dosages, frequencies, and indications from clinical notes. Include the prescribing physician if mentioned."

# ❌ Avoid: Vague prompts
bad_prompt = "Extract medical information"

# Use the good prompt
result = lx.extract(
    text_or_documents=clinical_notes,
    prompt_description=good_prompt,
    examples=good_examples,
    model_id="gemini-2.0-flash-exp"
)
```

### 3. Model Selection Guidelines

```python
# Choose model based on task complexity
def select_model(task_complexity: str) -> str:
    model_map = {
        "simple": "gemini-2.0-flash-exp",      # Fast, cost-effective
        "moderate": "gemini-2.0-flash-exp",    # Good balance
        "complex": "gemini-2.0-pro",           # Deep reasoning
        "specialized": "gemini-2.0-pro"        # Domain expertise
    }
    return model_map.get(task_complexity, "gemini-2.0-flash-exp")

# Usage
model_id = select_model("moderate")
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id=model_id
)
```

### 4. Output Quality Assurance

```python
def ensure_output_quality(result: lx.ExtractionResult) -> bool:
    """Ensure extraction output meets quality standards"""
    quality_checks = {
        "has_extractions": len(result.extractions) > 0,
        "has_source_spans": all(
            len(ext.source_spans) > 0 for ext in result.extractions
        ),
        "data_not_empty": all(
            ext.data for ext in result.extractions
        )
    }
    
    passed_checks = sum(quality_checks.values())
    total_checks = len(quality_checks)
    
    print(f"Quality score: {passed_checks}/{total_checks}")
    return passed_checks == total_checks

# Usage
result = lx.extract(...)
if ensure_output_quality(result):
    print("High quality extraction")
else:
    print("Consider refining examples or prompt")
```

## Troubleshooting

### Common Issues

1. **Empty Extractions**
   ```python
   # Check input text length and examples
   if not result.extractions:
       print(f"Input length: {len(text)} characters")
       print(f"Number of examples: {len(examples)}")
       # Try simpler examples or clearer prompt
   ```

2. **Inconsistent Output Format**
   ```python
   # Ensure examples follow consistent schema
   # Use more specific prompt descriptions
   # Consider using fewer but higher-quality examples
   ```

3. **Missing Source Spans**
   ```python
   # Verify text preprocessing doesn't remove character positions
   # Check if extraction entities exist in source text
   ```

4. **API Rate Limits**
   ```python
   # Implement exponential backoff
   # Use caching for repeated requests
   # Consider batch processing
   ```

### Debugging Checklist

- [ ] Examples follow consistent format
- [ ] Prompt is specific and actionable
- [ ] Input text is well-formatted
- [ ] Model selection matches task complexity
- [ ] API credentials are properly configured
- [ ] Rate limiting is implemented for production use

---

*For the latest updates and detailed documentation, visit the [LangExtract GitHub repository](https://github.com/google/langextract).*