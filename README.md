# Vietnamese Invoice OCR & Extraction

This project focuses on Optical Character Recognition (OCR) and Information Extraction (IE) for Vietnamese invoices. It combines **PaddleOCR** for text detection, **VietOCR** for text recognition, and **Google Gemini (LLM)** to parse unstructured text into structured JSON data (Universal Schema).

> **Note:** This is a hybrid pipeline project. It leverages the strengths of specialized OCR models for reading Vietnamese text and the reasoning capabilities of Large Language Models (Gemini 2.5 Flash) to understand document layout and semantics.

# Outline

1. Text Detection
2. Text Recognition
3. Information Extraction (LLM)
4. Usage

# 1. Text Detection

Text detection is the process of locating text in an image and recognizing the presence of characters. The [DB algorithm](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_db_en.md) is a popular algorithm used in the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) framework to localize text in the input invoice image. It works by detecting the text regions in the image and then grouping them into text lines. This algorithm is known for its high accuracy and speed.

To enhance the accuracy of Text Recognition, images cropped by the DB algorithm are padded to ensure that the text is not cut off during the recognition process.

# 2. Text Recognition

Text Recognition is the process of recognizing the text in an image. For this part, I utilized [VietOCR](https://github.com/pbcquoc/vietocr), a popular framework for Vietnamese OCR tasks based on Transformer architecture. The Transformer OCR architecture combines CNN (to extract features) and Transformer models (seq2seq) to recognize text. This architecture is particularly effective for Vietnamese language peculiarities.

# 3. Information Extraction (LLM)

OCR engines only provide raw, unstructured text. To convert this into usable data, this project integrates **Google Gemini API** (Gemini 2.5 Flash).

* **Role:** The LLM acts as a reasoning engine. It receives the raw text sorted by position, analyzes the context, and extracts key fields.
* **Capabilities:**
    * **Auto-Classification:** Distinguishes between Header, Body (Items), and Footer.
    * **Universal Schema:** Automatically maps data to a standardized JSON structure (Seller, Buyer, Invoice Info, Line Items, Financials).
    * **Error Correction:** Fixes minor OCR typos based on semantic context.
    * **Complex Layouts:** Handles invoices with tables, merged columns, or non-standard formats.

# 4. Usage

### Prerequisites

* Python 3.10+
* Google AI Studio API Key

### Installation

Firstly, clone this repository:

```bash
git clone [https://github.com/HienVuu/VietOCR.git](https://github.com/HienVuu/VietOCR.git)
cd VietOCR
It is recommended to use conda to manage the environment:
# Create and activate environment
conda create -n Vietocr_env python=3.10
conda activate Vietocr_env

# Install dependencies
pip install -r requirements.txt
Configuration
Set up your Google API Key (Required for Gemini integration):
Windows (PowerShell):
$env:GOOGLE_API_KEY="AIzaSy_YOUR_API_KEY_HERE"
Linux/macOS:
export GOOGLE_API_KEY="AIzaSy_YOUR_API_KEY_HERE"
Running Inference
You can run the extraction script for a single image or a whole directory.

Option 1: Process a single image
python run.py --image_path input/invoice_example.jpg --output_path output
Option 2: Process all images in a folder (Batch Processing)
python run.py --image_path input --output_path output
The results (images with bounding boxes and structured JSON files) will be saved in the output/ directory.

Jupyter Notebook
For experimentation and visualization, you can explore the code at predict.ipynb or inference.ipynb.

References
PaddleOCR

VietOCR

Google AI Studio (Gemini)