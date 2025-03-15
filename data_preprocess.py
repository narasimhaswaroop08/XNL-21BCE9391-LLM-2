import json
from datasets import Dataset

# Load SQuAD v2.0 dataset
with open("dev-v2.0.json", "r") as f:
    squad_data = json.load(f)

# Convert to T5 format
train_data = []
for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]

            # Handle unanswerable questions
            if qa.get("is_impossible", False) or not qa["answers"]:
                answer = "unanswerable"
            else:
                answer = qa["answers"][0]["text"]

            train_data.append({
                "input_text": f"question: {question} context: {context}",
                "target_text": answer
            })

# Create Hugging Face dataset
dataset = Dataset.from_list(train_data)
dataset.save_to_disk("t5_train_data")
