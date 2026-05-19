# Retail Shelf Intelligence Pipeline

## Architecture Diagram

The overall pipeline architecture is shown below:

![Architecture Diagram](architecture_diagram.png)

---

## Overview

This project is a prototype retail shelf intelligence pipeline developed as part of the ParallelDots Machine Learning Engineer assignment.

The system processes retail shelf images and generates business insights such as:

- Product detection
- OCR-based brand recognition
- Shelf analytics
- Structured JSON outputs
- Annotated visualizations

The pipeline was designed using lightweight pretrained models with a modular structure for scalability and experimentation.

---

## Features

- Product/Object Detection using YOLOv8
- OCR-based text extraction using EasyOCR
- Brand classification using keyword matching
- Multi-image batch processing
- Annotated image generation
- JSON-based business metrics generation
- Modular and extensible project structure

---

## Pipeline Flow

```text
Input Shelf Image
        ↓
YOLOv8 Product Detection
        ↓
Product Cropping
        ↓
EasyOCR Text Extraction
        ↓
Keyword-Based Brand Classification
        ↓
Business Metrics Generation
        ↓
Annotated Output + JSON Results
```

---

## Models and Tools Used

| Task | Model / Tool |
|------|---------------|
| Object Detection | YOLOv8n |
| OCR | EasyOCR |
| Image Processing | OpenCV |
| Classification | OCR + Keyword Matching |

---

## Why These Models?

- YOLOv8n was selected because it is lightweight and fast for CPU inference.
- EasyOCR provided a simple OCR pipeline without requiring custom training.
- Keyword-based matching allowed flexible brand recognition without retraining classification models.
- The overall focus was on building a practical inference pipeline rather than training large custom models.

---

## Project Structure

```text
parallel_dots_assignment/
│
├── classification/
│   └── brand_classifier.py
│
├── detection/
│   └── detect_products.py
│
├── utils/
│   └── save_results.py
│
├── ocr/
│
├── segmentation/
│
├── data/
│   ├── input/
│   └── output/
│
├── requirements.txt
├── architecture_diagram.png
├── README.md
└── yolov8n.pt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python -m classification.brand_classifier
```

---

## Output

The pipeline generates:

### Annotated Images
- Product bounding boxes
- OCR-based brand labels

### JSON Results
- Total detected products
- Brand counts
- OCR labels

### Example JSON Output

```json
{
    "image_name": "shelf_1.jpg",
    "total_products": 39,
    "brands": {
        "Coca-Cola": 4,
        "Fanta": 4,
        "Sprite": 3
    },
    "ocr_labels": [
        "Coca-Cola",
        "Fanta",
        "Sprite"
    ]
}
```

---

## Limitations

- Generic YOLO models may miss densely packed products.
- OCR quality decreases for blurry or partially visible products.
- Some products may be classified as `"Other"` due to OCR noise.
- Keyword matching is heuristic-based and depends on OCR quality.

---

## Future Improvements

- Custom retail-product detection model
- Better OCR preprocessing
- Instance segmentation for crowded shelves
- Database/API integration
- Real-time inference support

---

## Conclusion

This project demonstrates a modular retail shelf intelligence pipeline combining computer vision, OCR, and lightweight analytics generation for practical retail AI workflows.