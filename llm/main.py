# using new openai sdk

import base64
import os
import csv
from openai import OpenAI


client = OpenAI(api_key="your-api-key")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def classify_image(image_path):
    base64_image = encode_image(image_path)
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a medical expert. This is a brain MRI image. "
                            "Classify the image into one of the following categories exactly: glioma, meningioma, pituitary, notumor. "
                            "Respond only with the label — one of: glioma, meningioma, pituitary, notumor. "
                            "Do not explain. Do not include any other text."
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
    )
    return response.output_text.strip().lower()


username = os.environ.get("USERNAME")

image_folder = fr"C:\Users\{username}\Documents\nl\ISI_dataset\test_2"

output_csv = "gpt4_results.csv"

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]

results = []
for image_file in image_files:
    print(f"Classifying: {image_file}")
    image_path = os.path.join(image_folder, image_file)
    # label = classify_image(image_path)
    label = "glioma"
    results.append({ "id": image_file, "label": label })

print(results)

# with open(output_csv, "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["id", "label"])
#     writer.writeheader()
#     writer.writerows(results)

print(f"Classification complete. Output saved to {output_csv}")