import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class DataPreparator:
    def __init__(self, source_lang, target_lang, file_path1, file_path2, max_length, percentage=100,
                 filter_by_length=True, test_size=0.2):
        """
        :param source_lang: the source language
        :param target_lang: the target language
        :param file_path1: the file path of the source language
        :param file_path2: the file path of the target language
        :param percentage: the percentage that define the size of the dataset to work on
        :param max_length: maximum length of sentence
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.max_length = max_length
        self.percentage = percentage
        self.filter_by_length = filter_by_length
        self.test_size = test_size
        # create a Language instance for both source and target languages
        self.input_lang = Lang(self.source_lang)
        self.output_lang = Lang(self.target_lang)

    def prepare(self):
        df = self.load_dataset()

        if self.filter_by_length:
            df = self.filter_df(df)
            print("Filtering sentences with max length < {} results in:".format(self.max_length))
            print("{} sentence for each language".format(df.shape[0]))

        # Create training and validation sets using an 80-20 split
        train_df, val_df = train_test_split(df, test_size=self.test_size)

        #  Iterates through the train_df rows for each row, implement the addSentence method on source and target
        #  sentences using their corresponding Language class
        for index, row in train_df.iterrows():
            self.input_lang.addSentence(row[0])
            self.output_lang.addSentence(row[1])

        val_df = self.get_multiple_translations(val_df)

        return {self.source_lang:self.input_lang, self.target_lang: self.output_lang}, train_df , val_df

    def read_lang_file(self, file_path, lang):
        """
        This method reads a specific language file which has each text on a single line and returns a list of all the
        sentences
        :param file_path: the file path of which we want to read
        :param lang: a string the represents the language of the sentences in the file
        :return: a list of sentences
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lang_lines = f.readlines()
        print("We have loaded {} sentences for {} lang".format(len(lang_lines), lang))
        return lang_lines

    def normalize_string(self, s):
        """
        This method takes a text and returns a normalized version of it.
        It first lowers and removes newlines at the end of the texts.
        Then it separate punctuations concatenated to words by creating a space between a word and the punctuation;
        This is done by using a regular expression to match the punctuation characters .!?,() and surround them by
        spaces.
        Finally collapse multiple spaces anywhere in the text.
        :param s: a string
        :return: normalized text
        """
        s = s.lower().rstrip()
        # s = re.sub(r"([.!?,])", r" \1", s)
        s = re.sub(r"([a-zA-Z]{2,})([.!?,()])", r"\1 \2", s)
        s = re.sub(r"([.!?,()])([a-zA-Z]{2,})", r"\1 \2", s)
        s = re.sub(r"([.!?,()])([.!?,()])", r"\1 \2", s)
        s = re.sub(r"([.!?,()])([.!?,()])", r"\1 \2", s)
        s = re.sub(' +', ' ', s)
        return s

    def load_dataset(self):
        """
        This method loads source and target languages files and returns a dataframe with each column as a language.
        Select percentage of rows in dataframe, and remove duplicates
        """
        source_lang_lines = self.read_lang_file(self.file_path1, self.source_lang)
        target_lang_lines = self.read_lang_file(self.file_path2, self.target_lang)
        df = pd.DataFrame({self.source_lang: source_lang_lines, self.target_lang: target_lang_lines})
        #     maybe shuffle first
        df = df.head(int(len(df) * (self.percentage / 100)))  # select percentage of rows in pandas dataframe
        df = df.applymap(self.normalize_string)
        if self.percentage != 100:
            print("Selecting {} % of the dataset results in:".format(self.percentage))
            print(
                "{} sentence for both {} and {} languages each".format(len(df[self.source_lang]), self.source_lang,
                                                                       self.target_lang))

        df = df[df[self.source_lang].map(len) > 2]
        # get the unique values (rows) -> remove duplicated translations if exist
        df_drop_duplicates = df.drop_duplicates()

        return df_drop_duplicates

    def filter_df(self, df):
        """
        This method filters out sentences with more than a certain nbr of words
        :param df:
        :return:
        """
        def len_(sen):
            return len(sen.split())

        return df[np.logical_and(df[self.source_lang].map(len_) < self.max_length,
                                 df[self.target_lang].map(len_) < self.max_length)]

    def get_multiple_translations(self, df):
        """
        This method transforms a dataframe which contain the parallel data where each source is aligned with a target
         sentence into source sentences with multiple translations if exists
        :param df: input dataframe
        :return: source sentences aligned with target references in a dataframe
        """
        d = {k: [a.split() for a in g[self.target_lang].tolist()] for k, g in df.groupby(self.source_lang)}
        source_sentences = list(d.keys())  # [r.split() for r in list(d.keys())]
        target_references = list(d.values())
        return pd.DataFrame(data={self.source_lang:source_sentences, self.target_lang:target_references})

    def transform_to_langs_pairs(self, df):
        """
         This method transforms the source and target language sentences stored in separate columns into a list of
         tuples.
         The 1st element is the source sentence and the second element is the target sentence in the target language
         If number of examples is set then take a chunk of the data
        :param df: dataframe
        :return: [(lang1,lang2),(lang1,lang2),...]
        """
        records = df.to_records(index=False)
        pairs = list(records)
        return pairs

class Lang(object):
    """
    A class for each language that is going to map each word in a language to a unique integer number.
    It has 3 dictionary data structures:
     word2int: maps each word to a unique integer
     int2word: maps an integer to a word
     word2count: maps a word to its total number in the corpus
    """
    def __init__(self, name):
        self.name = name
        self.word2int = {"<SOS>": 1, "<EOS>": 2, "<UNK>": 3}  # maps words to integers
        #  EOS means End of Sentence and it's a token used to indicate the end of a sentence. Every sentence is going to
        #  have an EOS token. SOS means Start of Sentence and is used to indicate the start of a sentence.)
        self.int2word = {1: "<SOS>",
                         2: "<EOS>",
                         3: "<UNK>"}  # maps integers to tokens (just the opposite of word2int but has some initial
        # values.
        #         leaving 0 for padding
        self.word2count = {}  # maps words to their total number in the corpus
        self.n_words = 3  # Initial number of tokens (<EOS> and <SOS> and <UNK>)

    def addWord(self, word):
        """
        This method adds words from a specific language corpus to its language class dictionaries.
        It adds a word as a key to the word2int dictionary with its value being a corresponding integer
        The opposite is done for the int2word dictionary
        :param word:
        :return:
        """
        if word not in self.word2int:
            self.word2int[word] = self.n_words + 1
            self.word2count[word] = 1
            self.int2word[self.n_words + 1] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        """
        This method iterate through each sentence and for each sentence, it splits the sentence into words and
        implements the addWord method on each word in each sentence
        :param sentence:
        :return:
        """
        for word in sentence.split(' '):
            self.addWord(word)
