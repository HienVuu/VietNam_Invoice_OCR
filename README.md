
# Vietnamese OCR with Structured Data Extraction

A comprehensive Optical Character Recognition (OCR) system specifically designed for Vietnamese text, combining state-of-the-art OCR engines with AI-powered structured data extraction for various document types including invoices, technical specifications, and engineering drawings.

## Features

- **Multi-Engine OCR**: Utilizes PaddleOCR for text detection and VietOCR for Vietnamese text recognition
- **AI-Powered Structuring**: Leverages Google Gemini AI to extract and structure data from documents
- **Document Type Support**: Handles invoices, receipts, technical specifications, datasheets, packing lists, and technical drawings
- **Flexible Deployment**: Supports both Docker containerization and local installation
- **Batch Processing**: Efficiently processes multiple images in a single run
- **Structured Output**: Generates annotated images with bounding boxes and comprehensive JSON data
- **Technical Drawing Analysis**: Specialized extraction for engineering drawings including metadata, dimensions, and specifications

## Prerequisites

- Python 3.10+ (for local installation)
- Docker and Docker Compose (for containerized deployment)
- Google AI Studio API Key (required for Gemini integration)

## Installation and Setup

### Option 1: Docker Deployment (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/HienVuu/VietOCR.git
cd VietOCR
```

2. Set your Google API Key:
```bash
export GOOGLE_API_KEY="your_actual_api_key_here"
```

3. Place input images in the `input/` directory.

4. Build and run with Docker Compose:
```bash
docker-compose up --build
```

For custom image processing, modify the `command` in `docker-compose.yml`:
```yaml
command: ["python", "run.py", "--image_paths", "/app/input/your_image.jpg", "/app/input/another_image.png", "--output_path", "/app/output"]
```

### Option 2: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/HienVuu/VietOCR.git
cd VietOCR
```

2. Create and activate a conda environment:
```bash
conda create -n vietocr_env python=3.10
conda activate vietocr_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create the `input/` directory and place your input images there.

5. Configure your Google API Key:
   - **Windows (PowerShell)**:
     ```powershell
     $env:GOOGLE_API_KEY="your_actual_api_key_here"
     ```
   - **Linux/macOS**:
     ```bash
     export GOOGLE_API_KEY="your_actual_api_key_here"
     ```

## Usage

### Running Inference

Process single or multiple images using the main script:

**Single Image Processing:**
```bash
python run.py --image_paths input/your_image.jpg --output_path output
```

**Batch Processing:**
```bash
python run.py --image_paths input/image1.jpg input/image2.png input/image3.jpeg --output_path output
```

### Output Description

The system generates two types of output files in the specified output directory:

1. **Annotated Images**: Original images with detected text regions highlighted by bounding boxes
2. **Structured JSON Files**: Comprehensive data extraction including:
   - Document type classification
   - Raw OCR text
   - Structured fields based on document type (invoice details, technical specifications, drawing metadata, etc.)

### Supported Document Types

- **Invoices & Receipts**: Extracts vendor info, dates, item details, totals, and financial data
- **Technical Specifications**: Captures material properties, dimensions, tolerances, and technical parameters
- **Engineering Drawings**: Extracts metadata, dimensions, tolerances, approval information, and technical notes
- **Datasheets & Catalogs**: Structures product specifications and technical data
- **Packing Lists**: Organizes shipment and inventory information

## Project Structure

```
vietnamese-ocr/
├── run.py                      # Main inference script
├── combined_simple.py          # Simplified processing pipeline
├── create_ground_truth.py      # Ground truth data creation
├── validate_ground_truth.py    # Validation utilities
├── categorize_images.py        # Image categorization
├── download_dataset.py         # Dataset downloading utilities
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Python dependencies
├── dataset/                    # Training/validation datasets
│   ├── easy/                  # Easy difficulty samples
│   └── hard/                  # Hard difficulty samples
├── input/                     # Input images directory
├── output/                    # Generated results
├── temp_output/               # Temporary processing files
├── evaluation_results/        # Evaluation metrics and reports
├── PaddleOCR/                 # PaddleOCR engine
├── vietocr/                   # VietOCR engine
└── VietnameseOcrCorrection/   # Correction utilities
```

## Configuration

The system uses VietOCR's VGG-Transformer model by default for Vietnamese text recognition. Model configurations can be adjusted in the source code for different accuracy-performance trade-offs.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PaddleOCR for robust text detection
- VietOCR for Vietnamese text recognition capabilities
- Google Gemini AI for intelligent data structuring
=======
# 4. Usage

## Option 1: Docker (Recommended)

### Prerequisites

* Docker and Docker Compose
* Google AI Studio API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/HienVuu/VietOCR.git
cd VietOCR
```

2. Set your Google API Key:
```bash
export GOOGLE_API_KEY="AIzaSy_YOUR_API_KEY_HERE"
```

3. Place your input images in the `input/` directory.

4. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The results will be saved in the `output/` directory.

### Custom Usage

To process specific images, modify the `command` in `docker-compose.yml`:
```yaml
command: ["python", "run.py", "--image_paths", "/app/input/your_image.jpg", "/app/input/another_image.png", "--output_path", "/app/output"]
```

## Option 2: Local Installation

### Prerequisites

* Python 3.10+
* Google AI Studio API Key

### Installation

Firstly, clone this repository:

```bash
git clone https://github.com/HienVuu/VietOCR.git
cd VietOCR
```

It is recommended to use conda to manage the environment:
```bash
# Create and activate environment
conda create -n Vietocr_env python=3.10
conda activate Vietocr_env

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set up your Google API Key (Required for Gemini integration):
```bash
# Windows (PowerShell):
$env:GOOGLE_API_KEY="AIzaSy_YOUR_API_KEY_HERE"

# Linux/macOS:
export GOOGLE_API_KEY="AIzaSy_YOUR_API_KEY_HERE"
```

### Running Inference

You can run the extraction script for a single image or a whole directory.

**Option 1: Process a single image**
```bash
python run.py --image_paths input/invoice_example.jpg --output_path output
```

**Option 2: Process all images in a folder (Batch Processing)**
```bash
python run.py --image_paths input/image1.jpg input/image2.png --output_path output
```

The results (images with bounding boxes and structured JSON files) will be saved in the `output/` directory.

### Jupyter Notebook

For experimentation and visualization, you can explore the code at `predict.ipynb` or `inference.ipynb`.
