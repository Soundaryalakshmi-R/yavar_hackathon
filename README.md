
# Context-Grounded Image Captioning with Visual and Metadata Input

This project implements a **context-aware image captioning system** using the BLIP-2 FLAN-T5-XL model. It generates both **concise** and **detailed** captions grounded in:
- Visual content of the image
- Structured metadata: section header, caption, above text, below text, footnote

---

## Project Structure

```
.
├── ingest.py                # Converts raw JSON (Concadia format) into per-image metadata .txt files
├── preprocess.py            # Combines images and metadata into dataset.json with "context_text"
├── train.py                 # Fine-tunes BLIP-2 FLAN-T5-XL on structured image-text pairs
├── caption_generator.py     # Generates concise and detailed captions from the fine-tuned model
├── eval_metric.py           # Evaluates generated captions using BLEU, ROUGE, and consistency
├── ui.py                    # Streamlit interface for user-driven caption generation
├── output_folder/           # Stores generated captions and evaluation results
├── data/
│   ├── img_folder/          # Folder of images (JPG/PNG)
│   └── metadata_folder/     # Folder of per-image .txt metadata files
└── preprocessed/
    └── dataset.json         # Unified dataset with image IDs and context text
```

---

## Dataset Details

### 1. Concadia Dataset (Base)

We used the [Concadia dataset](https://github.com/vis-nlp/Concadia), a diverse corpus of document images paired with structured context, including:
- Scientific figures
- News graphics
- Medical illustrations
- Diagrams from various domains

**Modifications**:
- The original JSON descriptions were remapped into a structured metadata format:
  - `section_header`
  - `above_text`
  - `caption`
  - `below_text`
  - `footnote`
- A unified `context_text` string was constructed from the above fields to train and condition the model.

### 2. Custom Mini Dataset (Added)

A small, hand-crafted dataset was added to improve generalization and handle non-natural image types such as:
- Tables
- Bar charts, line graphs, pie charts
- Circuit diagrams
- Flowcharts and engineering blueprints
- Logos and scanned forms

This helped the model better handle structured and technical images with high metadata dependency.

---

## Approach

### Preprocessing
- `ingest.py` parses the original dataset and stores structured metadata as `.txt` files.
- `preprocess.py` reads image-metadata pairs and constructs a unified JSON (`dataset.json`) for training and inference.

### Training
- Model: `Salesforce/blip2-flan-t5-xl`
- Both concise and detailed captions are generated in one pass.
- Trained on GPU where available, with fallback to CPU.

### Inference
- `caption_generator.py` loads the trained model and processes images to generate:
  - Concise caption (short, visual description)
  - Detailed caption (expanded with metadata context)
- Each caption is annotated with a confidence score and warnings if confidence is low.

### Evaluation
- `eval_metric.py` computes:
  - BLEU score
  - ROUGE-L
  - BLIP self-consistency score (via sentence embedding similarity)

### Streamlit Interface
- Users can upload an image and optional metadata.
- If no metadata is provided, the model defaults to image-only captioning.
- Concise and detailed captions are visually displayed with length and confidence difference.
- <p align="center">
  <img src="image%20(2).png" width="600" height="400"/>
    </p>
 <p align="center">
  <img src="image%20(3).png" width="600" height="400"/>
  </p>
  <p align="center">
  <img src="image%20(4).png" width="600" height="400"/>
</p>






---

## Features

- Dual caption generation: concise and detailed
- Contextual grounding in structured metadata
- Visual confidence-based annotations
- Evaluation via BLEU, ROUGE-L, and embedding consistency
- Compatible with both CPU and GPU environments
- Interactive Streamlit UI for real-time testing

---

## Requirements

Install dependencies using pip:

```bash
pip install torch torchvision transformers sentence-transformers rouge-score nltk streamlit
```
