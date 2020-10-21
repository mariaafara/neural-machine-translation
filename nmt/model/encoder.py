import tensorflow as tf


class Encoder(tf.keras.models.Model):
    GRU_UNIT = 1
    LSTM_UNIT = 2

    def __init__(self, vocab_size, embedding_dim=256, encoder_size=256, batch_size=16, unit_type=GRU_UNIT):
        super(Encoder, self).__init__()
        # define model's layers here
        self.unit_type = unit_type
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="encoder_embedding")

        if self.unit_type == Encoder.GRU_UNIT:
            self.rnn = tf.keras.layers.GRU(self.encoder_size,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform', name="encoder_GRU")
        elif self.unit_type == Encoder.LSTM_UNIT:
            self.rnn = tf.keras.layers.LSTM(self.encoder_size,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform',
                                            name="encoder_LSTM")
        else:
            raise ValueError("Unknown unit type: {}".format(unit_type))

    def call(self, sequences, initial_state):
        """
        implement the model's forward pass here
        :param sequences: is of shape (batch_size, max_sequence length)
        :param initial_state: only for hidden state in case of GRU (batch_size, encoder_size) or one for hidden state and one for cell state in case of LSTM  2*(batch_size, encoder_size)
        it is a list of initial state tensors to be passed to the first call of the cell
        :return: rnn_out is of shape (batch_size, max_sequence length, encoder_size), both forward_hidden and forward_cell are of shapes (batch_size, encoder_size)
        """
        embedded_sequences = self.embedding(sequences)  # (batch_size, max_sequence length, vocab_size)
        if self.unit_type == Encoder.GRU_UNIT:
            rnn_output, forward_hidden = self.rnn(embedded_sequences, initial_state=initial_state)
            return rnn_output, forward_hidden
        elif self.unit_type == Encoder.LSTM_UNIT:
            rnn_output, forward_hidden, forward_cell = self.rnn(embedded_sequences, initial_state=initial_state)
            return rnn_output, forward_hidden, forward_cell

    def initialize_hidden_state(self):
        """
        creation of zero-filled initial state tensors
        """
        if self.unit_type == Encoder.GRU_UNIT:
            return tf.zeros(shape=(self.batch_size, self.encoder_size))
        elif self.unit_type == Encoder.LSTM_UNIT:
            return [tf.zeros(shape=(self.batch_size, self.encoder_size)) for i in range(2)]  # one for each state

