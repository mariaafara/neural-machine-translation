import tensorflow as tf


class Decoder(tf.keras.models.Model):

    GRU_UNIT = 1
    LSTM_UNIT = 2

    def __init__(self, vocab_size, embedding_dim=256, decoder_size=256, unit_type=GRU_UNIT):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_size = decoder_size
        self.unit_type = unit_type
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="decoder_embedding")
        if self.unit_type == Decoder.LSTM_UNIT:
            self.rnn = tf.keras.layers.LSTM(self.decoder_size,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform',
                                            name="decoder_LSTM ")
        elif self.unit_type == Decoder.GRU_UNIT:
            self.rnn = tf.keras.layers.GRU(self.decoder_size,
                                           recurrent_initializer='glorot_uniform',
                                           return_sequences=True,
                                           return_state=True,
                                           name="decoder_GRU")
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, initial_states):
        """
        the final states of the encoder will act as the initial states of the decoder
        :param sequence: (batch_size, max_sequence length)
        :param initial_states: initial_state of the decoder; if encoder was LSTM_UNIT: 2 * (batch_size, encoder_size)
        if GRU_UNIT: (batch_size, encoder_size)
        :return rnn_out is of shape (batch_size, max_sequence length, encoder_size), both forward_hidden and forward_cell are of shapes (batch_size, encoder_size)
        decoder_output: (batch_size, max_sequence length, vocab_size)
        """
        embedded_sequence = self.embedding(sequence) # (batch_size, max_sequence length, vocab_size)
        if self.unit_type == Decoder.GRU_UNIT:
            rnn_output, forward_hidden = self.rnn(embedded_sequence, initial_state=initial_states)
            decoder_output = self.dense(rnn_output)
            return decoder_output, forward_hidden
        elif self.unit_type == Decoder.LSTM_UNIT:
            rnn_output, forward_hidden, forward_cell = self.rnn(embedded_sequences, initial_state=initial_states)
            decoder_output = self.dense(rnn_output)
            return decoder_output, forward_hidden, forward_cell