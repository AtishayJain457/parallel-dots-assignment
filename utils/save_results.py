import json
import os

def save_results(
    image_name,
    total_products,
    brand_counts,
    ocr_labels,
    output_path="data/output/results.json"
):

    results = {
        "image_name": image_name,
        "total_products": total_products,
        "brands": brand_counts,
        "ocr_labels": ocr_labels
    }

    os.makedirs("data/output", exist_ok=True)

    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("\nJSON results saved successfully!")
    print(f"Saved at: {output_path}")