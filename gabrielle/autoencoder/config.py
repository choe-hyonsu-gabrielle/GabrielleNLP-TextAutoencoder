import dataclasses


@dataclasses.dataclass
class TextAutoConfig:
    MAX_LENGTH = 250
    EMBEDDING_DIM = 512
    HIDDEN_SIZE = 512
    BATCH_SIZE = 64
    EPOCHS = 100
    SAVED_MODEL_PATH = 'saved_model'
    DOCUMENTS = 'C:\\Users\\choeh\\PycharmProjects\\GabrielleNLP-TextAutoencoder\\data\\sentences.txt'
