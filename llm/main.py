# using new openai sdk

import base64
import os
import csv
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

if __name__ == "__main__":

    username = os.environ.get("USERNAME")
    image_folder = Path(fr"C:\Users\{username}\Documents\nl\ISI_dataset\test_2")

    output_csv = "llm_chatgpt_gpt4dot1_results.csv"

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()

        for entry in image_folder.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".jpg":
                print(f"Classifying: {entry.name}")
                label = classify_image(str(entry))
                json_value = {"id": entry.name, "label": label}
                print(json_value)
                writer.writerow(json_value)
                f.flush()

    print(f"Classification complete. Output saved to {output_csv}")