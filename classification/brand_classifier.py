from utils.save_results import save_results
from ultralytics import YOLO
import easyocr
import cv2
import os

# LOAD PRETRAINED YOLO MODEL

model = YOLO("yolov8n.pt")

# INITIALIZE OCR READER
reader = easyocr.Reader(['en'])

# BRAND KEYWORDS
brand_keywords = {

    "Coca-Cola": [
        "coca",
        "coke",
        "cocacola",
        "coca cola",
        "cola"
    ],

    "Pepsi": [
        "pepsi"
    ],

    "Sprite": [
        "sprite"
    ],

    "Fanta": [
        "fanta"
    ],

    "7UP": [
        "7up",
        "7 up",
        "up"
    ],

    "Thums Up": [
        "thums",
        "thumbs",
        "thumsup",
        "thums up",
        "thump"
    ],

    "Mirinda": [
        "mirinda",
        "mirnda",
        "minda"
    ],

    "Mountain Dew": [
        "dew",
        "mountain"
    ],

    "Tropicana": [
        "tropicana"
    ],

    "Real": [
        "real"
    ],

    "Minute Maid": [
        "minute"
    ],

    "Paper Boat": [
        "paperboat",
        "paper boat"
    ],

    "Red Bull": [
        "redbull",
        "red bull"
    ],

    "Gatorade": [
        "gatorade"
    ],

    "Lays": [
        "lays"
    ],

    "Doritos": [
        "doritos"
    ],

    "Oreo": [
        "oreo"
    ],

    "Kurkure": [
        "kurkure"
    ],

    "Amul": [
        "amul"
    ]
}

# CREATE OUTPUT DIRECTORY
os.makedirs("data/output", exist_ok=True)

# PROCESS ALL INPUT IMAGES
for image_name in os.listdir("data/input"):

    # Skip non-image files
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print("\n===================================")
    print(f"PROCESSING IMAGE: {image_name}")
    print("===================================\n")

    # FULL IMAGE PATH
    image_path = os.path.join("data/input", image_name)

    # READ IMAGE
    image = cv2.imread(image_path)

    # RUN YOLO DETECTION
    
    results = model(image_path)

    # STORE BRAND COUNTS
    brand_counts = {}

    # STORE UNIQUE OCR LABELS
    ocr_labels = set()

  
    total_products = 0

 
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # CONFIDENCE SCORE
            confidence = float(box.conf[0])

            # Skip weak detections
            if confidence < 0.50:
                continue
            # BOUNDING BOX COORDINATES
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # FILTER VERY SMALL DETECTIONS
            width = x2 - x1
            height = y2 - y1

            if width < 40 or height < 40:
                continue

            total_products += 1

            
            crop = image[y1:y2, x1:x2]

            # Skip invalid crop
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, None, fx=2, fy=2)

            -
            ocr_results = reader.readtext(crop)

            detected_text = ""

            # Combine OCR text
            for detection in ocr_results:
                detected_text += detection[1] + " "

            
            # CLEANING OCR TEXT FOR BETTER READING / UNDERSTANDING
            
            detected_text = detected_text.lower().strip()
            detected_text = detected_text.replace("\n", " ")

            
            # DEBUG OCR OUTPUT
            
           
            print("OCR Text:", detected_text)

            
            # DEFAULT BRAND
            
            detected_brand = "Other"

            
            # CHECKING KEYWORD ARE MATCHING
            
            for brand, keywords in brand_keywords.items():

                found = False

                for keyword in keywords:

                    if keyword in detected_text:
                        detected_brand = brand
                        found = True
                        break

                if found:
                    break

            -
            if detected_brand != "Other":
                ocr_labels.add(detected_brand)

           
            # UPDATING  BRAND COUNTS
            
            if detected_brand not in brand_counts:
                brand_counts[detected_brand] = 0

            brand_counts[detected_brand] += 1

           
            # DRAW BOUNDING BOX
            
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1
            )

          
            label = f"{detected_brand}"

           
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            # labelling the background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 8),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )

            # putting text on label upper side
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

# image path
    output_path = f"data/output/brand_detection_{image_name}"

# saving o/p name
    cv2.imwrite(output_path, image)

   
    print("\n===================================")
    print(" BRAND DETECTION COMPLETED ")
    print("===================================\n")

    print(f"Total Products Detected: {total_products}\n")

    print("Brand Counts:")
    for brand, count in brand_counts.items():
        print(f"{brand}: {count}")

    print("\nOutput Image Saved At:")
    print(output_path)

   # here converting sets to list
    ocr_labels = list(ocr_labels)

 # saving result in json
    save_results(
        image_name=image_name,
        total_products=total_products,
        brand_counts=brand_counts,
        ocr_labels=ocr_labels,
        output_path=f"data/output/results_{os.path.splitext(image_name)[0]}.json"
    )