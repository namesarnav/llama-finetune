from config import MODEL_ID, HF_TOKEN, TRAIN_FILE, TEST_FILE, FINAL_MODEL_DIR
from models.model_loader import load_model, load_tokenizer
from models.peft_setup import get_lora_model
from data.dataset_loader import preprocess_data
from utils.metrics import compute_metrics

from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, EarlyStoppingCallback


def main():
    print("Loading model + tokenizer...")
    model = load_model(MODEL_ID, HF_TOKEN)
    tokenizer = load_tokenizer(MODEL_ID, HF_TOKEN)
    if getattr(model.config, "vocab_size", None) and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id # you need this because the error is not with the tokenizer, but with the model itself
    
    print("Preprocessing dataset...")

    tk_data_train, tk_data_test, _ = preprocess_data(tokenizer, TRAIN_FILE, TEST_FILE)

    # resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    print("Preparing LoRA model...")
    model = get_lora_model(model)

    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=250,
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tk_data_train,
        eval_dataset=tk_data_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("Evaluating...")
    results = trainer.evaluate(tk_data_test)
    print(f"Final results: {results}")


if __name__ == "__main__":
    main()
