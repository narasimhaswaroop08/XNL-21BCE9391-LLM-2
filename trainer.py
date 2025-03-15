import os
import json
import torch
import deepspeed
import optuna
import csv
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW, SGD
from sklearn.metrics import accuracy_score, f1_score
import mlflow  # Import MLflow

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# DeepSpeed configuration
deepspeed_config = {
    "train_batch_size": "auto",  # Set to "auto" to match TrainingArguments
    "train_micro_batch_size_per_gpu": "auto",  # Set to "auto" to match per_device_train_batch_size
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}

# Save DeepSpeed config to a file
with open("ds_config.json", "w") as f:
    json.dump(deepspeed_config, f)

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load preprocessed datasets
train_dataset = load_from_disk("t2")  # Training dataset
val_dataset = load_from_disk("v2")  # Validation dataset

# Preprocess dataset
def preprocess_function(examples):
    model_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Optuna objective function
def objective(trial):
    # Start an MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "batch_size": trial.suggest_categorical("batch_size", [2, 4]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
            "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        })

        # Hyperparameters to tune
        batch_size = trial.params["batch_size"]
        learning_rate = trial.params["learning_rate"]
        optimizer_name = trial.params["optimizer"]
        warmup_steps = trial.params["warmup_steps"]
        weight_decay = trial.params["weight_decay"]

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./t5_finetuned_trial_{trial.number}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            save_strategy="epoch",  # Save at the end of each epoch
            logging_dir=f"./logs_trial_{trial.number}",
            logging_steps=10,
            num_train_epochs=1,  # Only 1 epoch
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            fp16=True,
            deepspeed="ds_config.json",
            save_total_limit=1,  # Keep only the latest checkpoint
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Create optimizer
        if optimizer_name == "adamw":
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, None),  # Pass the optimizer explicitly
        )

        # Learning rate scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(trainer.get_train_dataloader()) * training_args.num_train_epochs,
        )
        trainer.lr_scheduler = scheduler

        # CSV file to store scores and metrics
        csv_file = f"./t5_finetuned_trial_{trial.number}/scores.csv"
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "eval_loss", "accuracy", "f1_score", "learning_rate", "batch_size", "optimizer", "warmup_steps", "weight_decay"])

        # Train and evaluate
        try:
            for epoch in range(training_args.num_train_epochs):  # Only 1 epoch
                print(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")
                trainer.train()

                # Evaluate on the validation set
                val_results = trainer.evaluate(val_dataset)
                val_loss = val_results["eval_loss"]
                val_predictions = trainer.predict(val_dataset)

                # Extract logits from the predictions tuple
                val_logits = val_predictions.predictions[0]  # Logits are the first element of the tuple

                # Get predicted token IDs using argmax
                val_preds = val_logits.argmax(axis=-1)

                # Extract labels
                val_labels = val_predictions.label_ids

                # Flatten predictions and labels
                val_preds_flat = val_preds.flatten()  # Flatten predictions
                val_labels_flat = val_labels.flatten()  # Flatten labels

                # Filter out padding tokens (if any)
                padding_token_id = 0  # Change if your tokenizer uses a different padding token ID
                non_padding_mask = val_labels_flat != padding_token_id

                # Apply the mask to filter out padding tokens
                val_preds_filtered = val_preds_flat[non_padding_mask]
                val_labels_filtered = val_labels_flat[non_padding_mask]

                # Calculate accuracy and F1 score
                val_accuracy = accuracy_score(val_labels_filtered, val_preds_filtered)
                val_f1 = f1_score(val_labels_filtered, val_preds_filtered, average="weighted")

                # Log metrics to MLflow
                mlflow.log_metrics({
                    "train_loss": val_results.get("loss", None),
                    "eval_loss": val_loss,
                    "accuracy": val_accuracy,
                    "f1_score": val_f1,
                })

                # Save training and validation scores
                epoch_dir = f"./t5_finetuned_trial_{trial.number}/epoch_{epoch + 1}"
                os.makedirs(epoch_dir, exist_ok=True)

                train_scores = {
                    "epoch": epoch + 1,
                    "train_loss": val_results.get("loss", None),
                    "learning_rate": val_results.get("learning_rate", None),
                }
                val_scores = {
                    "epoch": epoch + 1,
                    "eval_loss": val_loss,
                    "accuracy": val_accuracy,
                    "f1_score": val_f1,
                }

                # Write scores to CSV
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        train_scores["train_loss"],
                        val_scores["eval_loss"],
                        val_scores["accuracy"],
                        val_scores["f1_score"],
                        train_scores["learning_rate"],
                        batch_size,
                        optimizer_name,
                        warmup_steps,
                        weight_decay,
                    ])

                # Save console logs
                log_file = f"{epoch_dir}/epoch_{epoch + 1}_logs.txt"
                with open(log_file, "w") as f:
                    f.write(f"Epoch: {epoch + 1}\n")
                    f.write(f"Training Loss: {train_scores['train_loss']}\n")
                    f.write(f"Validation Loss: {val_scores['eval_loss']}\n")
                    f.write(f"Validation Accuracy: {val_scores['accuracy']}\n")
                    f.write(f"Validation F1 Score: {val_scores['f1_score']}\n")
                    f.write(f"Learning Rate: {train_scores['learning_rate']}\n")

                # Print scores to console
                print(f"Epoch: {epoch + 1}")
                print(f"Training Loss: {train_scores['train_loss']}")
                print(f"Validation Loss: {val_scores['eval_loss']}")
                print(f"Validation Accuracy: {val_scores['accuracy']}")
                print(f"Validation F1 Score: {val_scores['f1_score']}")
                print(f"Learning Rate: {train_scores['learning_rate']}")

                # Save the model in its original format
                model_dir = f"./models/trial_{trial.number}_epoch_{epoch + 1}"
                os.makedirs(model_dir, exist_ok=True)
                model.save_pretrained(model_dir)  # Saves in original format
                tokenizer.save_pretrained(model_dir)

                # Log the model directory as an artifact
                mlflow.log_artifacts(model_dir, artifact_path="model")

        except Exception as e:
            print(f"Training interrupted due to: {e}")
            print("Saving files before exiting...")
            # Ensure files are saved even if an error occurs
            model_dir = f"./models/trial_{trial.number}_epoch_{epoch + 1}"
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            raise e  # Re-raise the exception after saving files

        # Final evaluation
        metrics = trainer.evaluate()
        return metrics["eval_loss"]

# Create an Optuna study
study = optuna.create_study(direction="minimize")

# Run hyperparameter optimization
study.optimize(objective, n_trials=10)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train with the best hyperparameters
best_trial = optuna.create_study(direction="minimize")
best_trial.enqueue_trial(study.best_params)
best_trial.optimize(objective, n_trials=1)
