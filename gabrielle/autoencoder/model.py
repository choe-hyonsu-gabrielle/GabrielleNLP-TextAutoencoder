import dataclasses
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Concatenate, Dense, TimeDistributed, Input


@dataclasses.dataclass
class TextAutoConfig:
    EMBEDDING_DIM = 512
    HIDDEN_SIZE = 256


class SharedEmbedding(layers.Layer):
    def __init__(self, vocab_size, embedding_dim=TextAutoConfig.EMBEDDING_DIM):
        super(SharedEmbedding, self).__init__(name=self.__class__.__name__)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)

    def call(self, inputs, *args, **kwargs):
        embedded = self.embedding(inputs)
        return embedded


class TextAutoEncoder(layers.Layer):
    def __init__(self, embedding_dim=TextAutoConfig.EMBEDDING_DIM, hidden_size=TextAutoConfig.HIDDEN_SIZE):
        super(TextAutoEncoder, self).__init__(name=self.__class__.__name__)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.gru_head = Bidirectional(GRU(embedding_dim//2, return_sequences=True, dropout=0.2))
        self.gru_tail = Bidirectional(GRU(hidden_size//2, return_state=True, dropout=0.2))

    def call(self, inputs, *args, **kwargs):
        # inputs = embedding outputs
        sequence_outputs = self.gru_head(inputs)
        _output, forward_state, backward_state = self.gru_tail(sequence_outputs)
        encoder_final_state = Concatenate(axis=-1)([forward_state, backward_state])
        return encoder_final_state


class TextAutoDecoder(layers.Layer):
    def __init__(self, vocab_size, hidden_size=TextAutoConfig.HIDDEN_SIZE):
        super(TextAutoDecoder, self).__init__(name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.gru_head = GRU(self.hidden_size, return_sequences=True)
        self.softmax = TimeDistributed(Dense(self.vocab_size, activation='softmax'))

    def call(self, inputs, *args, **kwargs):
        # inputs = embedding outputs
        sequence_outputs = self.gru_head(inputs[0], initial_state=inputs[1])
        decoder_outputs = self.softmax(sequence_outputs)
        return decoder_outputs


def get_autoencoder_model(max_length, vocab_size):
    encoder_input = Input(shape=(max_length,), dtype='int32')
    decoder_input = Input(shape=(max_length,), dtype='int32')
    shared_embedder = SharedEmbedding(vocab_size=vocab_size)
    encoder = TextAutoEncoder()
    decoder = TextAutoDecoder(vocab_size=vocab_size)

    encoder_embedding = shared_embedder(encoder_input)
    encoder_final_state = encoder(encoder_embedding)
    decoder_embedding = shared_embedder(decoder_input)
    decoder_output = decoder([decoder_embedding, encoder_final_state])

    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


if __name__ == '__main__':
    model = get_autoencoder_model(50, 3000)
