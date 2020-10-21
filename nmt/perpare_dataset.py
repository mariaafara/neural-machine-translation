import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_lang_file(file_path, lang):
    """
    This method reads a specific language file which has each text on a single line and returns a list of all the sentences
    :param file_path: the file path of which we want to read
    :param lang: a string the represents the language of the sentences in the file
    :return: a list of sentences
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lang_lines = f.readlines()
    print("Total sentences of {} lang = {}".format(lang, len(lang_lines)))
    return lang_lines


def normalize_string(s):
    """
    This method takes a text and returns a normalized version of it.
    It first lowers and removes newlines at the end of the texts.
    Then it separate punctuations concatenated to words by creating a space between a word and the punctuation;
    This is done by using a regular expression to match the punctuation characters .!?, and surround them by spaces.
    Finally collapse multiple spaces anywhere in the text.
    :param s: a string
    :return: normalized text
    """
    s = s.lower().rstrip()
    s = re.sub(r"([.!?,])", r" \1", s)
    s = re.sub(' +', ' ', s)
    return s


def load_dataset(source_lang, target_lang, file_path1, file_path2, percentage=100):
    """
    This method loads source and target languages files and returns a dataframe with each column as a language.
    :param source_lang: the source language
    :param target_lang: the target language
    :param file_path1: the file path of the source language
    :param file_path2: the file path of the target language
    :param percentage: the percentage that define the size of the dataset to work on
    :return:
    """
    source_lang_lines = read_lang_file(file_path1, source_lang)
    target_lang_lines = read_lang_file(file_path2, target_lang)
    df = pd.DataFrame({source_lang: source_lang_lines, target_lang: target_lang_lines})
    #     maybe shuffle first
    df = df.head(int(len(df) * (percentage / 100)))  # select percentage of rows in pandas dataframe
    df = df.applymap(normalize_string)
    print("selected number of sentences of {} lang is = {}".format(source_lang, len(df[source_lang])))
    print("selected number of sentences of {} lang is = {}".format(target_lang, len(df[target_lang])))
    return df


def transform_to_langs_pairs(df, number_examples=None):
    """
     This method transforms the source and target language sentences stored in separate columns into a list of tuples.
     The 1st element is the source sentence and the second element is the target sentence in the target language
     If number of examples is set then take a chunk of the data
    :param df: dataframe
    :param number_examples: number of examples to take
    :return: [(lang1,lang2),(lang1,lang2),...]
    """
    records = df[:number_examples].to_records(index=False)
    list_pairs = list(records)
    return list_pairs


def filter_pairs(langs_pairs, MAX_LENGTH):
    """
    This method filters out sentences with more than a certain nbr of words
    :param langs_pairs: pairs of sentences [(source_lang sentence, target_lang sentence),..]
    :param MAX_LENGTH: maximum length of sentence wanted
    :return: filtered list of pairs
    """
    def filter_pair(langs_pair, MAX_LENGTH):
        return len(langs_pair[0].split()) < MAX_LENGTH and \
            len(langs_pair[1].split()) < MAX_LENGTH
    return [lang_pair for lang_pair in langs_pairs if filter_pair(lang_pair, MAX_LENGTH)]


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
        self.word2int = {"<SOS>": 1, "<EOS>": 2}  # maps words to integers
        #  EOS means End of Sentence and it's a token used to indicate the end of a sentence. Every sentence is going to have an EOS token. SOS means Start of Sentence and is used to indicate the start of a sentence.)
        self.int2word = {1: "<SOS>", 2: "<EOS>"}  # maps integers to tokens (just the opposite of word2int but has some initial values.
#         leaving 0 for padding
        self.word2count = {}  # maps words to their total number in the corpus
        self.n_words = 2  # Initial number of tokens (<EOS> and <SOS>)

    def addWord(self, word):
        """
        This method adds words from a spe
        cific language corpus to its language class dictionaries.
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
        This method iterate through each sentence and for each sentence, it splits the sentence into words and implements the addWord method on each word in each sentence
        :param sentence:
        :return:
        """
        for word in sentence.split(' '):
            self.addWord(word)


def sentence_to_indexes(sentence, lang, EOS_token=2):
    """
    This method encodes a sentence according to its language by giving an index to each word in the sentence
    It Iterates through a sentence, breaks it into words and maps each word to its corresponding integer value using the word2int dictionary which was implemented in the language class
    Finally append the EOS_token (end of sentence)
    :param sentence:
    :param lang: the language instance corresponding to the sentence (e.g. for an French sentence, we use the French language instance)
    :param EOS_token:
    :return: encoded sentence with numerical values
    """
    indexes = [lang.word2int[word] for word in sentence.split()]
    indexes.append(EOS_token) #TODO: check later
    return indexes


#  and assign a corresponding  integer to each word in a new function.
# Also since we are going to batch our dataset, we are going to apply padding to sentences with words less than the maximum length we proposed.

def build_lang(lang1, lang2, pairs, MAX_LENGTH):
    """
    This method creates a Language class for both source and target languages
    :param lang1: source language
    :param lang2:  target language
    :param pairs: list of pairs : [(sentence in lang2, sentence in lang2)...]
    :param MAX_LENGTH: max length of sequence
    :return: padded sequences
    """
    # this function populates the word2int dictionary in each langauge class and pads
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    input_seq = []
    output_seq = []

    #  Iterates through the list of pairs and for each pair, implement the addSentence method on source and target sentences using their corresponding Language class
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    #  Convert each sentence in each pair into numerical values using the sentence_to_indexes method
    for pair in pairs:
        input_seq.append(sentence_to_indexes(pair[0], input_lang))
        output_seq.append(sentence_to_indexes(pair[1], output_lang))

    #  Pads sequences to the same length.
    #  we assign the argument post to pad or truncate at the end of the sequence
    #  it takes a list of sequences (each sequence is a list of integers) and returns a 2D Numpy array with shape (len(sequences), maxlen)
    return pad_sequences(input_seq, maxlen=MAX_LENGTH, padding='post', truncating='post'), pad_sequences(output_seq, padding='post', truncating='post'), input_lang, output_lang


def sort_into_batches(input_tensor, output_tensor, BATCH_SIZE):
    """
    This method sorts the dataset into batches to accelerate our training process using TensorFlow data API
    :param input_tensor:
    :param output_tensor:
    :param BATCH_SIZE:
    :return:
    """
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset
