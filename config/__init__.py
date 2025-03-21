from pydantic_settings import BaseSettings


class Config(BaseSettings):

    MODEL_VOCAB_SIZE: int = 20000
    MODEL_EMBEDDING_DIM: int = 256
    MODEL_HIDDEN_DIM: int = 128
    MODEL_NUM_LAYERS: int = 2
    MODEL_DROPOUT: float = 0.3

    TRAINING_BATCH_SIZE: int = 64
    TRAINING_LEARNING_RATE: float = 1e-4
    TRAINING_EPOCHS: int = 15
    TRAINING_GRAD_CLIP: float = 1.0

    DATA_MAX_SEQ_LENGTH: int = 256
    DATA_MIN_WORD_FREQ: int = 5
