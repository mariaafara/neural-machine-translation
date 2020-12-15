import tensorflow as tf


class BahdanauAttention(tf.keras.models.Model):
    """
    BahdanauAttention also known as Additive Attention calculates additive/concat alignment score which quantifies how
     much attention we should give to each input
    """
    def __init__(self, decoder_size):
        super(BahdanauAttention, self).__init__()
        # decoder_size == decoder_units
        self.W1 = tf.keras.layers.Dense(decoder_size)
        self.W2 = tf.keras.layers.Dense(decoder_size)
        self.V = tf.keras.layers.Dense(1)

    def call(self, encoder_out, decoder_hidden):
        """
        The forward propagation takes place here.
        At each time step of the decoder, we have to calculate the alignment score of each encoder output with respect
        to the decoder input and hidden state at that time step.
        The score quantifies the amount of “Attention” the decoder will place on each of the encoder outputs when
        producing the next output.
         Steps:
        - The decoder hidden state and encoder outputs will be passed through their individual Linear layer and have their
          own individual trainable weights.
        - Then, they will be added together before being passed through a tanh activation function.
        - The decoder hidden state is added to each encoder output in this case.
        - The resultant vector will undergo matrix multiplication with a trainable vector, obtaining a final alignment
          score vector which holds a score for each encoder output.
        - Then apply a softmax on this vector to obtain the attention weights. The softmax function will cause the
          values in the vector to sum up to 1 and each individual value will lie between 0 and 1, therefore representing
          the weightage each input holds at that time step.
        - Finally, generate the context vector by doing an element-wise multiplication of the attention weights with
        the encoder outputs
        - The context vector that is produced will then be concatenated with the previous decoder output.
          It is then fed into the decoder RNN cell to produce a new hidden state and the process keeps repeating itself.

        :param encoder_out: Encoder's all hidden states at each timestep,
         shape = (batch_size, max_sequence length, hidden_size)
        :param decoder_hidden: the hidden state produced by the decoder in the previous time step before the current timestep
         where we are generating the correct word,
         shape = (batch_size, hidden_size)
        :return: context vector and the attention weights
        """
        # we are doing this to broadcast addition along the time axis to calculate the score
        decoder_hidden = tf.expand_dims(decoder_hidden, axis=1) # (batch_size, 1, hidden size)

        # score shape = (batch_size, max_sequence length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_sequence length, decoder_size)
        score = self.V(tf.nn.tanh(self.W1(decoder_hidden) +
                                  self.W2(encoder_out))) # (batch_size, max_sequence length, 1)

        attn_weights = tf.nn.softmax(score, axis=1) # (batch_size, max_sequence length, 1)

        context = attn_weights * encoder_out
        context = tf.reduce_sum(context, axis=1) # (batch_size, hidden_size)
        return context, attn_weights