import math
import numpy as np


TOTAL_NUMBER_OF_DOCUMENTS = 2500


def read_documents():
    docs = []
    for i in range(1, TOTAL_NUMBER_OF_DOCUMENTS + 1):
        document_name = 'documents/document%s.txt' % i
        words_in_doc = list_of_words(document_name)
        docs.append(words_in_doc)
    return docs


def list_of_words(document):
    list = []
    f = open(document)
    for line in f:
        words = process_line(line)
        list.extend(words)
    f.close()
    return list


def process_line(line):
    all_words = line.strip().split(' ')
    words = remove_invalid_word(all_words)
    return words


def remove_invalid_word(words):
    return [word for word in words if valid_word(word)]


def valid_word(word):
    return len(word) > 0 and word.isalnum()


def add_one(dictionary, word):
    dictionary[word] = 1


def get_idf(total_documents, occurrences):
    idf_dict = dict()
    for key, value in occurrences.iteritems():
        inverse_freq = float(total_documents) / float(1 + value)
        idf_dict[key] = math.log(inverse_freq)

    values = idf_dict.values()
    return np.diag(values).astype(np.longdouble)


def get_empty_tf_matrix(rows, words):
    tf_formats = ['longdouble'] * len(words)
    return np.zeros((rows, len(words)), dtype={'names': words, 'formats': tf_formats})


if __name__ == '__main__':
    docs = read_documents()

    occurrences = dict()
    for doc in docs:
        map(lambda w: add_one(occurrences, w), doc)

    idf = get_idf(TOTAL_NUMBER_OF_DOCUMENTS, occurrences)
    tf = get_empty_tf_matrix(TOTAL_NUMBER_OF_DOCUMENTS, occurrences.keys())

    print tf.shape
    print idf.shape

    tf_idf = np.dot(tf.astype(np.longdouble), idf)
    print tf_idf.shape