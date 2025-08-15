# sentenial-x/ai_core/config.py

MODEL_PATHS = {
    "bert_threat_classifier": "./models/bert_intent_classifier",
    "lora_adapters": "./models/lora",
    "distilled_models": "./models/distill",
    "text_encoder": "./models/encoder",
}

EMBEDDING_DIM = 256
THREAT_SCORE_THRESHOLD = 0.6   # score above which a log is flagged
JAILBREAK_SCORE_THRESHOLD = 0.7

DEVICE = "cuda"  # switch to "cpu" if GPU not available
BATCH_SIZE = 16
MAX_SEQ_LEN = 128
