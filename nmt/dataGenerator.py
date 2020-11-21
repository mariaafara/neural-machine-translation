import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generators are a way to provide batches of data on-the-fly during training phase.
    It is most needed when dealing with huge datasets, where the whole data cannot be stored in memory.
    """
    def __init__(self, lang_dic, df, max_length, batch_size, phase_train=True):
        self.lang_dic = lang_dic
        self.df = df
        self.max_length = max_length
        self.batch_size = batch_size
        self.size = df.shape[0]
        self.phase_train = phase_train
        self.columns = df.columns.tolist()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.size // self.batch_size

    def on_epoch_end(self):
        'Updates indexes after each epoch and shuffle'
        self.indexes = np.arange(self.size)
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch`
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch = self.df.iloc[indexes]
        if self.phase_train:
            batched_input_seq, batched_output_seq = self.encode_sentence(self.lang_dic[self.columns[0]],
                                                                         self.lang_dic[self.columns[1]], batch)
            batched_padded_input_seq, batched_padded_output_seq = self.padd(batched_input_seq, batched_output_seq)
        return batched_padded_input_seq, batched_padded_output_seq

    def sentence_to_indexes(self, sentence, lang, EOS_token=2):
        """
        This method encodes a sentence according to its language by giving an index to each word in the sentence
        It Iterates through a sentence, breaks it into words and maps each word to its corresponding integer value using
         the word2int dictionary which was implemented in the language class
        Finally append the EOS_token (end of sentence)
        :param sentence:
        :param lang: the language instance corresponding to the sentence (e.g. for an French sentence, we use the French
         language instance)
        :param EOS_token:
        :return: encoded sentence with numerical values
        """
        indexes = [lang.word2int[word] if word in lang.word2int else lang.word2int["<UNK>"] for word in
                   sentence.split()]
        indexes.append(EOS_token)  # TODO: check later
        return indexes

    def encode_sentence(self, input_lang, output_lang, df):
        #  Convert each sentence in each pair into numerical values using the sentence_to_indexes method
        input_seq = []
        output_seq = []
        for index, row in df.iterrows():
            input_seq.append(self.sentence_to_indexes(row[0], input_lang))
            output_seq.append(self.sentence_to_indexes(row[1], output_lang))
        # df_new = pd.DataFrame()
        # df_new["input"] = df.apply(lambda row : self.sentence_to_indexes(row[0], input_lang), axis=1)
        # df_new["output"] = df.apply(lambda row : self.sentence_to_indexes(row[1], output_lang), axis=1)
        #
        return input_seq, output_seq

    def padd(self, input_seq, output_seq):
        """
         This method pads sequences to the same length.
        :param input_seq:
        :param output_seq:
        :param df:
        :return: padded sequences for src and target
        """

        #  we assign the argument post to pad or truncate at the end of the sequence
        #  it takes a list of sequences (each sequence is a list of integers) and returns a 2D Numpy array with shape
        #  (len(sequences), maxlen)
        # df_new = pd.DataFrame()
        # pad_src = pad_sequences(df["input"].tolist(), maxlen=self.max_length, padding='post', truncating='post')
        # # with or without specifying the max len for the target pad it will pad to the max length of the sentences
        # pad_target = pad_sequences(df["output"].tolist(), padding='post', truncating='post')
        # df_new["input"] = pad_src
        # df_new["output"] = pad_target
        # return df_new

        pad_src = pad_sequences(input_seq, maxlen=self.max_length, padding='post', truncating='post')
        # with or without specifying the max len for the target pad it will pad to the max length of the sentences
        pad_target = pad_sequences(output_seq, padding='post', truncating='post')

        return pad_src, pad_target
