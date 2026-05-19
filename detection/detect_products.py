from ultralytics import YOLO
import cv2
import os


# LOAD PRETRAINED YOLO MODEL

model = YOLO("yolov8n.pt")

# INPUT IMAGE PATH

image_path = "data/input/shelf_1.jpg"


results = model(image_path)


# READ IMAGE USING OPENCV

image = cv2.imread(image_path)


for result in results:

    boxes = result.boxes

    for box in boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confidence score
        confidence = float(box.conf[0])

        # FILTER LOW CONFIDENCE DETECTIONS
        if confidence < 0.50:
            continue

        # Class ID
        class_id = int(box.cls[0])

        # Class name
        class_name = model.names[class_id]

        # Label text
        label = f"{class_name}: {confidence:.2f}"

       
        # DRAW BOUNDING BOX
        
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1
        )

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        
        # DRAW LABEL BACKGROUND
        
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 8),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1
        )

       
        # PUT LABEL TEXT
        
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )


# CREATE OUTPUT DIRECTORY

os.makedirs("data/output", exist_ok=True)


# OUTPUT IMAGE PATH

output_path = "data/output/output_shelf_1.jpg"


# SAVE OUTPUT IMAGE

cv2.imwrite(output_path, image)

print("\nDetection completed successfully!")
print(f"Output saved at: {output_path}")