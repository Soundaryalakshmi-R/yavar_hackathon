<<<<<<< HEAD
import os
import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from config import config
from dataset import BLIP2Dataset
from torch.utils.data import DataLoader

# # Force CPU by hiding GPU from PyTorch
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# device = torch.device("cpu")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Force CPU if GPU is full
if torch.cuda.is_available():
    try:
        torch.empty((1024, 1024), device="cuda")  # test allocation
    except RuntimeError:
        print("GPU out of memory, falling back to CPU.")
        device = torch.device("cpu")


# Collatorimport os
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# Main
def main():
    processor = Blip2Processor.from_pretrained(config.PROCESSOR_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.float32
    ).to(device)

    # Dataset (limit for CPU demo)
    dataset = BLIP2Dataset(
        json_path=config.DATASET_JSON,
        img_folder=config.IMAGE_FOLDER,
        processor=processor,
        max_samples=200,  # keep small for CPU
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    # Custom Trainer
    class BLIP2Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            pixel_values = inputs.get("pixel_values").to(device)
            input_ids = inputs.get("input_ids").to(device)
            attention_mask = inputs.get("attention_mask").to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            return outputs.loss if not return_outputs else (outputs.loss, outputs)

    # Initialize trainer
    trainer = BLIP2Trainer(
        model=model, args=training_args, train_dataset=dataset, data_collator=collate_fn
    )

    # Train
    print("Starting training on CPU...")
    trainer.train()

    # Save
    model.save_pretrained(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)
    print(f"Model saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
=======
import os
import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from config import config
from dataset import BLIP2Dataset
from torch.utils.data import DataLoader

# # Force CPU by hiding GPU from PyTorch
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# device = torch.device("cpu")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Force CPU if GPU is full
if torch.cuda.is_available():
    try:
        torch.empty((1024, 1024), device="cuda")  # test allocation
    except RuntimeError:
        print("GPU out of memory, falling back to CPU.")
        device = torch.device("cpu")


# Collatorimport os
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# Main
def main():
    processor = Blip2Processor.from_pretrained(config.PROCESSOR_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.float32
    ).to(device)

    # Dataset (limit for CPU demo)
    dataset = BLIP2Dataset(
        json_path=config.DATASET_JSON,
        img_folder=config.IMAGE_FOLDER,
        processor=processor,
        max_samples=200,  # keep small for CPU
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    # Custom Trainer
    class BLIP2Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            pixel_values = inputs.get("pixel_values").to(device)
            input_ids = inputs.get("input_ids").to(device)
            attention_mask = inputs.get("attention_mask").to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            return outputs.loss if not return_outputs else (outputs.loss, outputs)

    # Initialize trainer
    trainer = BLIP2Trainer(
        model=model, args=training_args, train_dataset=dataset, data_collator=collate_fn
    )

    # Train
    print("Starting training on CPU...")
    trainer.train()

    # Save
    model.save_pretrained(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)
    print(f"Model saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
>>>>>>> master
