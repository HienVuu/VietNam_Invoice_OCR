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
