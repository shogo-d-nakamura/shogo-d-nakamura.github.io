---
layout: post
title: translating-pdfs-without-breaking-layout-using-plamo-translate-macos
date: 2025-06-08 10:33 +0900
description: ''
category: ''
tags: [PLaMo, LLM, PDFMathTranslate]
published: true
lang: en
ref: plamo_pdf_translate
---

# PLaMo Integrated PDFMathTranslate

For papers I want to read quickly, Japanese is preferable, so I had Claude Code create this tool.\
When you input an English PDF, it outputs a Japanese PDF and a bilingual (Japanese/English) PDF.\
It runs PLaMo translate locally using mlx-lm.\
This is for Mac, but it works on other OSes by changing the inference part using mlx-lm.\
Memory usage is around 6-10 GB.\
Tested on macOS Sequoia 15.4.1, M4 MacBook with 16 GB memory.

![Example](/assets/img/2025_images/translate_example.png)

Source code:\
[https://github.com/shogo-d-nakamura/PDF-PLaMoTranslator](https://github.com/shogo-d-nakamura/PDF-PLaMoTranslator)

Examples are shown in run.sh.\
Create a venv environment with `zsh install.sh`, download PLaMo translate with `python DL_plamo_translate.py`, then specify the PDF or directory containing PDFs you want to translate with -i as shown in run.sh.\
When specifying a PDF file (article.pdf), it outputs article-out.pdf and article-dual.pdf.\
When specifying a directory (articles), it creates articles-out/ and sequentially outputs {$PDF_FILENAME}-out.pdf and {$PDF_FILENAME}-dual.pdf.

```zsh
# for PDF
python translate.py -i ~/workspace/PLaMoTranslator/article.pdf

# for batch translation
python translate.py -i ~/workspace/PLaMoTranslator/articles
```

Below is the Japanese translation of the README created by Claude.

## Quick Start

### 1. Install Dependencies

```zsh
# Create virtual environment and install PDFMathTranslate and MLX-LM
zsh install.sh
```

### 2. Download Model
```zsh
python DL_plamo_translate.py
```

### 3. Translate PDF

```zsh
# Test sample
python translate_test_pdf.py
```

This creates `test-out.pdf` and `test-dual.pdf`.

## Features

### PLaMo Integration Features
- **Smart Token Management**: Automatically chunks text into approximately 1024 tokens
- **Sentence Boundary Preservation**: Maintains sentence boundaries during chunking
- **High-Quality Translation**: Uses mlx-community/plamo-2-translate model
- **Layout Preservation**: Maintains original PDF format using PDFMathTranslate

### Automatic Features
- **Token Estimation**: Estimates approximately 4 characters per token
- **Error Handling**: Proper fallback for translation failures
- **Progress Tracking**: Shows translation progress for large documents
- **Format Support**: Handles mathematical formulas and complex layouts

## Usage Examples

### Basic Translation Script

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'PDFMathTranslate')
from pdf2zh import translate

# Translate PDF using PLaMo
translate(
    files=["input.pdf"],
    output="./output",
    lang_in="en",        # English input
    lang_out="ja",       # Japanese output
    service="plamo"      # Use PLaMo translation
)
```

### Command Line Usage

```zsh
# Navigate to PDFMathTranslate directory
cd PDFMathTranslate

# Translate using PLaMo service
python -m pdf2zh.cli \
    --service plamo \
    --lang-in en \
    --lang-out ja \
    --output ../output \
    ../test.pdf
```

### Advanced Usage

```python
from pdf2zh import translate

# Custom configuration
translate(
    files=["document.pdf"],
    output="./translations",
    lang_in="en",
    lang_out="ja", 
    service="plamo:custom-model-name",  # Use custom model
    thread=1,                           # Single thread for PLaMo
    ignore_cache=True,                  # Fresh translation
    envs={
        "PLAMO_MODEL": "mlx-community/plamo-2-translate",
        "PLAMO_MAX_TOKENS": "1024"
    }
)
```

## Configuration

### Environment Variables

Set the following environment variables to customize PLaMo behavior:

```zsh
export PLAMO_MODEL="mlx-community/plamo-2-translate"
export PLAMO_MAX_TOKENS="1024"
```

### Model Configuration

```python
# Use different PLaMo models
service_options = [
    "plamo",                                    # Default model
    "plamo:mlx-community/plamo-2-translate",   # Explicit model specification
    "plamo:custom-plamo-model"                 # Custom model
]
```

## How It Works

### Token Management Process

1. **Text Extraction**: Extract text from PDF pages
2. **Token Estimation**: Estimate token count (approximately 4 characters = 1 token)
3. **Smart Chunking**: Split text at sentence boundaries
4. **Translation**: Translate each chunk using PLaMo via MLX-LM
5. **Reconstruction**: Recombine translated text while preserving layout