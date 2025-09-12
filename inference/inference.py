from transformers import pipeline
from torch import load
import pyyaml


model = load(MODEL_DIR)