import os
import json
import requests
import torch
import optuna
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Set ngrok authtoken (replace with your actual ngrok token)
os.environ["NGROK_AUTHTOKEN"] = "2uJxaqpvO7Grxnd2e66xGIJG17m_5jeqqXkwBBEDvB7cdyKMA"
conf.get_default().auth_token = os.environ["NGROK_AUTHTOKEN"]

# Start ngrok tunnel for external communication
public_url = ngrok.connect(5000, "http").public_url
print(f"Public Webhook URL for n8n: {public_url}")

# n8n webhook URLs
N8N_TRIGGER_URL = "https://orbitaipyramid.app.n8n.cloud/webhook-test/webhook/start-training"  # Replace with your n8n trigger URL
N8N_RESULTS_URL = "https://orbitaipyramid.app.n8n.cloud/webhook-test/webhook/results"  # Replace with your n8n results URL

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Create a custom dataset with one example
custom_data = {
    "input_text": ["Translate English to French: The house is wonderful."],
    "target_text": ["La maison est magnifique."]
}

# Convert the custom data to a Hugging Face Dataset
dataset = Dataset.from_dict(custom_data)

# Preprocess dataset
def preprocess_function(examples):
    model_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

# Global hyperparameters
hyperparams = {
    "batch_size": 8,
    "learning_rate": 3e-4,
    "optimizer": "adamw",
    "warmup_steps": 200,
    "weight_decay": 1e-5
}

def send_ngrok_url_to_n8n():
    """Send the ngrok URL to n8n"""
    try:
        response = requests.post(N8N_TRIGGER_URL, json={"ngrok_url": public_url})
        if response.status_code == 200:
            print("Successfully sent ngrok URL to n8n")
        else:
            print(f"Failed to send ngrok URL to n8n. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending ngrok URL to n8n: {e}")

def fetch_hyperparameters():
    """Fetch hyperparameters from n8n"""
    try:
        response = requests.get(f"{public_url}/webhook/hyperparameters")
        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to fetch hyperparameters from n8n, using defaults.")
            return hyperparams
    except Exception as e:
        print(f"Error fetching hyperparameters: {e}")
        return hyperparams

def send_results(results):
    """Send training results back to n8n"""
    try:
        response = requests.post(N8N_RESULTS_URL, json=results)
        if response.status_code == 200:
            print("Successfully sent results to n8n")
        else:
            print(f"Failed to send results to n8n. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending results: {e}")

def objective(trial):
    """Optuna objective function for hyperparameter tuning"""
    hyperparams = fetch_hyperparameters()

    batch_size = hyperparams.get("batch_size", trial.suggest_categorical("batch_size", [4, 8, 16]))
    learning_rate = hyperparams.get("learning_rate", trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True))
    optimizer = hyperparams.get("optimizer", trial.suggest_categorical("optimizer", ["adamw", "sgd", "lamb"]))
    warmup_steps = hyperparams.get("warmup_steps", trial.suggest_int("warmup_steps", 0, 500))
    weight_decay = hyperparams.get("weight_decay", trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))

    training_args = TrainingArguments(
        output_dir="./t5_finetuned",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        logging_dir="./logs",
        logging_steps=5,
        num_train_epochs=3,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        fp16=True,  # Mixed precision training
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Send training metrics to n8n
    send_results({
        "eval_loss": metrics["eval_loss"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay
    })

    return metrics["eval_loss"]

# Flask app for webhook communication
app = Flask(__name__)

@app.route('/webhook/hyperparameters', methods=['GET', 'POST'])
def handle_hyperparameters():
    """Endpoint for n8n to fetch or update hyperparameters"""
    if request.method == 'GET':
        return jsonify({"status": "success", "hyperparams": hyperparams})
    elif request.method == 'POST':
        data = request.json
        hyperparams.update(data)
        print(f"Updated hyperparameters: {hyperparams}")
        return jsonify({"status": "success", "message": "Hyperparameters updated successfully", "hyperparams": hyperparams})

@app.route('/webhook/results', methods=['POST'])
def receive_results():
    """Endpoint to receive training results from Colab"""
    data = request.json
    print(f"Received Training Results: {data}")
    return jsonify({"status": "success", "message": "Results received successfully", "received": data})

@app.route('/webhook/start-training', methods=['POST'])
def start_training():
    """Endpoint for n8n to trigger a new training cycle"""
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best_hyperparams = study.best_params
    send_results(best_hyperparams)

    return jsonify({"status": "success", "message": "Training completed", "best_hyperparams": best_hyperparams})

# Send ngrok URL to n8n at startup
send_ngrok_url_to_n8n()

# Start Flask server
if __name__ == "__main__":
    app.run(port=5000)
