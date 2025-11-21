# Vietnamese OCR - AI Agent Instructions

## Project Overview

A two-stage OCR pipeline for Vietnamese text recognition:
1. **Text Detection** (PaddleOCR): Locates text regions using DB (Differentiable Binarization) algorithm
2. **Text Recognition** (VietOCR): Recognizes Vietnamese characters using CNN+Transformer architecture

Key entry point: `run.py` combines both pipelines with configurable CPU/GPU execution.

## Architecture & Components

### Core Pipeline (`run.py`)
- **Detector**: `PaddleOCR` instance processes full image → outputs text bounding boxes
- **Recognizer**: `VietOCR.Predictor` processes cropped regions → outputs character sequences
- **Workflow**: Image → Detection (boxes) → Recognition (texts) → Visualization → Save annotated image
- **Padding Logic**: Crops padded by 4px (configurable) to prevent text cutoff during recognition

### Framework Directories

**PaddleOCR/** - Text detection models
- `paddleocr.py`: Main API (auto-downloads models to `~/.paddleocr/`)
- `tools/infer/predict_det.py`: Detection inference pipeline
- Config YAMLs in `configs/det/` - supports English, multilingual (lang='en' or 'ml')
- Imports via: `from PaddleOCR import PaddleOCR, draw_ocr`

**vietocr/** - Vietnamese text recognition  
- `tool/predictor.py`: `Predictor` class loads pre-trained weights, runs inference
- `tool/config.py`: `Cfg` loads YAML configs by name (e.g., `load_config_from_name('vgg_transformer')`)
- Supports beam search decoding (configurable in YAML `predictor.beamsearch`)
- Imports via: `from vietocr.vietocr.tool.predictor import Predictor` and `from vietocr.vietocr.tool.config import Cfg`

**VietnameseOcrCorrection/** - Optional text post-processing (Vietnamese grammar correction)
- Not currently integrated in `run.py` (marked as TODO)
- `config.py`: Maps Vietnamese accented characters (á, à, ả, etc.) for error correction
- Can be plugged in after recognition for spelling/grammar fixes

### Data Flow
```
Image File
  ↓ (cv2.imread)
PaddleOCR.ocr() → Bounding Boxes ([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
  ↓ (PIL crop with padding)
Cropped Regions
  ↓ 
VietOCR.Predictor.predict() → Text Strings
  ↓ (draw_ocr visualization)
Annotated Image → Save to output_path
```

## Dependencies & Environment

**Critical NumPy Issue**: This project has a known compatibility issue with **NumPy 2.0+**:
- `imgaug` library (required by PaddleOCR) uses deprecated `np.sctypes` 
- **Fix**: Always pin `numpy<2.0` when installing dependencies
- Command: `pip install "numpy<2.0"`

**Minimal Dependencies** (from `requirement.txt`):
```
einops==0.2.0
pyclipper
lmdb
```

**Implicit Dependencies** (not in requirements):
- `paddle` (PaddleOCR framework)
- `torch` (VietOCR model loading)
- `opencv-python` (cv2)
- `Pillow` (PIL)
- `imgaug` (PaddleOCR data augmentation - **NumPy 2.0 incompatible**)

**Model Downloads**:
- PaddleOCR: Auto-downloads to `~/.paddleocr/` on first run (English: `en_PP-OCRv3_det_infer`, `en_PP-OCRv3_rec_infer`)
- VietOCR: Downloads from URL in config or uses local weights file

## Project-Specific Patterns

### 1. Two-Model Initialization Pattern
Models initialized separately, not as pipeline:
```python
detector = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'  # Override config device
recognitor = Predictor(config)  # Note: typo in variable name (recognitor not recognizer)
```
**Convention**: Device handling differs - PaddleOCR uses `use_gpu` param, VietOCR uses config override.

### 2. Configuration Loading
VietOCR configs loaded by name from `vietocr/config/`:
- `vgg_transformer` → `vietocr/config/vgg-transformer.yml`
- `vgg_seq2seq` → `vietocr/config/vgg-seq2seq.yml`
- Config structure: YAML with `predictor`, `transformer`, `dataset`, `trainer` sections
- Always override `device` field after loading: `config['device'] = 'cpu'` or `'cuda'`

### 3. Bounding Box Format
PaddleOCR returns boxes as lists of 4 points: `[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]` (quadrilateral)
- Access via: `box[0]` (top-left), `box[2]` (bottom-right)
- Crop coordinates: `(int(box[0][0]-padding), int(box[0][1]-padding), int(box[2][0]+padding), int(box[2][1]+padding))`

### 4. Image Format Handling
- **Input**: File paths (string), processed by `cv2.imread()` or `PIL.Image.open()`
- **VietOCR expects**: PIL Image objects
- **Output**: NumPy arrays (from `draw_ocr()`), converted back to PIL for DPI handling

### 5. Font Path Handling
- Font must be TTF file (e.g., `./PaddleOCR/doc/fonts/latin.ttf`)
- `draw_ocr()` requires absolute or relative path from execution directory
- Common error: Font file not found if paths are relative and CWD differs

## Development Workflows

### Running Inference
```powershell
python run.py --image_path hoa_don.jpg --output_path ./output
```

**Key Arguments**:
- `--image_path`: Required, path to input image (must exist)
- `--output_path`: Optional, default `./output`, auto-creates directory

### Adding Features
- **New detection config**: Add YAML to `PaddleOCR/configs/det/`
- **New recognition model**: Update config in `vietocr/config/` and override in `run.py`
- **Post-processing**: Add after `texts = recognitor.predict()` loop or integrate `VietnameseOcrCorrection.train.py` module

### Testing with Notebooks
- `predict.ipynb`: Contains original development workflow (cells not executed)
- `colab_paddle.ipynb`: Google Colab variant with hardware acceleration
- VietnameseOcrCorrection/`inference.ipynb`: Text correction examples

## Common Issues & Fixes

**Issue**: `AttributeError: np.sctypes was removed in NumPy 2.0`
- **Root**: `imgaug` library incompatibility
- **Fix**: `pip install "numpy<2.0"`

**Issue**: `FileNotFoundError: Font file not found`
- **Check**: Font path relative to current working directory
- **Fix**: Use absolute paths or verify `FONT = './PaddleOCR/doc/fonts/latin.ttf'` exists from script's CWD

**Issue**: `ConnectionError: Model download failed`
- **Cause**: PaddleOCR/VietOCR attempting to download models
- **Fix**: Check internet, or pre-download and set local paths in config

## Code Style Notes

- **Language**: Vietnamese comments in source code (inline and docstrings)
- **Variable Naming**: Non-English ("recognitor" instead of "recognizer", "bbox" instead of "boxes")
- **Imports**: Relative imports within PaddleOCR framework, absolute imports in entry scripts
- **Modular Design**: Frameworks bundled locally (not pip packages), enables custom modifications

## Key Files for Modification

| File | Purpose | Modification Pattern |
|------|---------|---------------------|
| `run.py` | Main inference entry | Add args, modify pipeline stages, tweak padding/DPI |
| `PaddleOCR/paddleocr.py` | Detector config | Change `lang`, `use_angle_cls`, `use_gpu` params |
| `vietocr/config/vgg-transformer.yml` | Recognition config | Override `device`, `predictor.beamsearch`, vocabulary |
| `VietnameseOcrCorrection/config.py` | Vietnamese grammar rules | Add character mappings for correction post-processing |

---

**Last Updated**: November 2025 | **Project Status**: Compilation of frameworks (end-to-end integration in progress)
