import math
import numpy as np


def get_occurrences():
    occurrences = dict()
    for i in range(1, total_number_of_documents + 1):
        document_name = 'documents/document%s.txt' % i
        process_document_for_occurrence(document_name, occurrences)
    return occurrences


def process_document_for_occurrence(document, dictionary):
    f = open(document)
    for line in f:
        words = process_line(line)
        map(lambda w: add_one(dictionary, w), words)
    f.close()


def process_line(line):
    all_words = line.strip().split(' ')
    words = remove_invalid_word(all_words)
    return words


def remove_invalid_word(words):
    return [word for word in words if valid_word(word)]


def valid_word(word):
    return len(word) > 0 and word.isalnum()


def add_one(dictionary, word):
    if word in dictionary:
        dictionary[word] += 1
    else:
        dictionary[word] = 1


def get_idf(occurrences):
    idf = dict()
    for key, value in occurrences.iteritems():
        inverse_freq = float(total_number_of_documents) / float(1 + value)
        idf[key] = math.log(inverse_freq)
    return idf


def get_empty_tf_matrix(rows, words):
    tf_formats = ['float64'] * len(words)
    return np.zeros((rows, len(words)), dtype={'names': words, 'formats': tf_formats})


if __name__ == '__main__':
    total_number_of_documents = 2500

    occurrences = get_occurrences()
    idf = get_idf(occurrences)
    list_of_words = idf.keys()

    for v in idf.values():
        assert type(v) is float

    tf = get_empty_tf_matrix(total_number_of_documents, list_of_words)

    for row in range(1, total_number_of_documents + 1):
        document_name = 'documents/document%s.txt' % row
        f = open(document_name)
        for line in f:
            words = process_line(line)
            for word in words:
                tf[row - 1][word] += 1
        f.close()
        print document_name

    print tf.shape
    print idf.shape
    # tf_idf = np.multiply(tf, idf)
    # print tf_idf
