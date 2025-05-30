from transformers import TrainingArguments


class Config:
    # Model config
    MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
    PROCESSOR_NAME = "Salesforce/blip2-flan-t5-xl"

    # Training config
    BATCH_SIZE = 1
    EPOCHS = 1
    LEARNING_RATE = 5e-5
    FP16 = True
    GRADIENT_ACCUMULATION_STEPS = 1

    # Paths
    OUTPUT_DIR = "./blip-finetuned-FLAN"
    DATASET_JSON = r"C:\Users\pssan\Desktop\sound\preprocessed1\dataset.json"
    IMAGE_FOLDER = r"C:\Users\pssan\Desktop\sound\data\img_folder"

    def get_training_args(self):
        return TrainingArguments(
            output_dir="./blip-captions",
            per_device_train_batch_size=1,  # safe on CPU
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",
        )


config = Config()
