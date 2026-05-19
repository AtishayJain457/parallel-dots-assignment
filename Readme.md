# Retail Shelf Intelligence Pipeline



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

- The pipeline uses a generic pretrained YOLOv8 model rather than a retail-specific detection model, which can reduce detection precision for densely packed shelves.

- OCR quality is highly dependent on product visibility, image resolution, text orientation, and lighting conditions.

- Some products are classified as `"Other"` when OCR text is noisy or insufficient for reliable keyword matching.

- The current classification approach uses lightweight heuristic keyword mapping instead of a dedicated trained brand-classification network.

- Shelf images containing overlapping or partially occluded products may reduce OCR and classification accuracy.

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
