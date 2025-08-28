import os

MODEL_ID = "meta-llama/llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "./final_model"
