from gabrielle.tokenizer import CharLevelTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from model import get_autoencoder_model

if __name__ == '__main__':
    tokenizer = CharLevelTokenizer().load_pretrained_tokenizer()
    with open('C:\\Users\\choeh\\PycharmProjects\\GabrielleNLP-TextAutoencoder\\data\\sentences.txt', encoding='utf-8') as f:
        data = f.read().splitlines()

    train_encoder_x = [x for x in tokenizer.encode(data)]
    train_decoder_x = [x[:-1] for x in tokenizer.encode(data)]
    train_decoder_y = [y[1:] for y in tokenizer.encode(data)]

    # print(tokenizer.decode(train_encoder_x[0]))
    # print(tokenizer.decode(train_decoder_x[0]))
    # print(tokenizer.decode(train_decoder_y[0]))
    # print('Length of longest sequence from source:', max([len(x) for x in train_encoder_x]))

    # padding with max_len
    max_length = 250
    train_encoder_input = pad_sequences(sequences=train_encoder_x, maxlen=max_length, padding='post')
    train_decoder_input = pad_sequences(sequences=train_decoder_x, maxlen=max_length, padding='post')
    train_decoder_output = pad_sequences(sequences=train_decoder_y, maxlen=max_length, padding='post')

    model = get_autoencoder_model(max_length, tokenizer.vocab_size)

    callbacks = [
        ModelCheckpoint('SavedModel', monitor='acc', mode='max', verbose=1, save_best_only=True),
        EarlyStopping(monitor='acc', mode='max', verbose=1, patience=2),
        TerminateOnNaN()
    ]

    model.fit(x=[train_encoder_input, train_decoder_input], y=train_decoder_output, batch_size=64, epochs=50, callbacks=callbacks)


