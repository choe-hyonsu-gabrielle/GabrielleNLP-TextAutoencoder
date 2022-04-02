import absl.logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from gabrielle.tokenizer import CharLevelTokenizer
from gabrielle.autoencoder.model import get_autoencoder_model
from gabrielle.autoencoder.config import TextAutoConfig

# ignore "WARNING:absl:Found untraced functions such as..."
absl.logging.set_verbosity(absl.logging.ERROR)

if __name__ == '__main__':
    # load data
    with open(TextAutoConfig.DOCUMENTS, encoding='utf-8') as f:
        data = f.read().splitlines()

    # tokenizing texts
    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer()
    train_encoder_x = [x for x in tokenizer.encode(data)]
    train_decoder_x = [x[:-1] for x in tokenizer.encode(data)]
    train_decoder_y = [y[1:] for y in tokenizer.encode(data)]

    """ test prints
    print(tokenizer.decode(train_encoder_x[0]))
    print(tokenizer.decode(train_decoder_x[0]))
    print(tokenizer.decode(train_decoder_y[0]))
    print('Length of longest sequence from source:', max([len(x) for x in train_encoder_x]))
    """

    # padding with max_len
    max_length = 250
    train_encoder_input = pad_sequences(sequences=train_encoder_x, maxlen=max_length, padding='post')
    train_decoder_input = pad_sequences(sequences=train_decoder_x, maxlen=max_length, padding='post')
    train_decoder_output = pad_sequences(sequences=train_decoder_y, maxlen=max_length, padding='post')

    # load compiled model & fit
    model = get_autoencoder_model(max_length, tokenizer.vocab_size)
    callbacks = [
        ModelCheckpoint(TextAutoConfig.SAVED_MODEL_PATH, monitor='acc', mode='max', verbose=1, save_best_only=True),
        EarlyStopping(monitor='acc', mode='max', verbose=1, patience=3),
        TerminateOnNaN()
    ]
    model.fit(x=[train_encoder_input, train_decoder_input], y=train_decoder_output, callbacks=callbacks,
              batch_size=TextAutoConfig.BATCH_SIZE, epochs=TextAutoConfig.EPOCHS)


