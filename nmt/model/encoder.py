import tensorflow as tf

class Encoder(tf.keras.models.Model):
    GRU_UNIT = 1
    LSTM_UNIT = 2
    def __init__(self, vocab_size, encoder_units=256, num_embedding=256, batch_size=16, unit_type=GRU_UNIT):
        super(Encoder, self).__init__()
        self.unit_type = unit_type
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.num_embedding = num_embedding
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_embedding)

        if self.unit_type == self.GRU_UNIT:
            self.rnn = tf.keras.layers.GRU(self.encoder_units,
                                       return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
        elif unit_type == Encoder.LSTM_UNIT:
            self.rnn = tf.keras.layers.LSTM(self.encoder_units,
                                       return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
        else:
            raise ValueError("Unknown unit type: {}".format(unit_type))


    def call(self, x, initial_state=None):
        embedded_x = self.embedding(x)
        if self.how == Encoder.UNIDIRECTIONAL:
          if self.unit_type == Encoder.GRU_UNIT:
             rnn_output, forward_hidden = self.rnn(embedded_x, initial_state=initial_state)
             return rnn_output, forward_hidden
          elif self.unit_type == Encoder.LSTM_UNIT:
            rnn_output, forward_hidden, forward_cell = self.rnn(embedded_x, initial_state=initial_state)
            return rnn_output, forward_hidden, forward_cell

