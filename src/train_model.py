from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
from pdf_processor import process_pdfs


def prepare_training_data(chunks):
    # Convert chunks to dataset format
    data = {"text": [chunk.page_content for chunk in chunks]}
    dataset = Dataset.from_dict(data)
    return dataset


def train_model(dataset, model_name="gpt2", output_dir="models/custom_llm"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


if __name__ == "__main__":
    chunks = process_pdfs()
    if chunks:
        dataset = prepare_training_data(chunks)
        train_model(dataset)