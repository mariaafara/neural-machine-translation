import pandas as pd
import re

def read_lang_file(file_path,lang):
    """
    This method reads a specific language file which has each text on a single line and returns a list of all the sentences
    :param file_path: the file path of which we want to read
    :param lang: a string the represents the language of the sentences in the file
    :return: a list of sentences
    """
    with open(file_path,'r',encoding='utf-8') as f:
        lang_lines = f.readlines()
    print("Total sentences of {} lang = {}".format(lang,len(lang_lines)))
    return lang_lines

def normalizeString(s):
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

# Limit the size of the dataset to experiment faster (optional)
# Training on the complete dataset of >100,000 sentences will take a long time.
# To train faster, we can limit the size of the dataset to a certain percentage assigned as a parameter (of course, translation quality degrades with less data)

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
    df = df.applymap(normalizeString)
    print("selected number of sentences of {} lang is = {}".format(source_lang, len(df[source_lang])))
    print("selected number of sentences of {} lang is = {}".format(target_lang, len(df[target_lang])))
    return df

def transform_to_langs_pairs(df,number_examples=None):
    """
     This method transforms the source and target language sentences stored in separate columns into a list of tuples.
     The 1st element is the source sentence and the second element is the target sentence in the target language
     If number of examples is set then take a chunk of the data
    :param df: dataframe
    :param number_examples: number of examples to take
    :return: [(lang1,lang2),(lang1,lang2),...]
    """
    records = df[:number_examples].to_records(index=False)
    list_pairs= list(records)
    return list_pairs
