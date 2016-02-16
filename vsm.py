import math
import collections
import numpy as np
import scipy.sparse as sps


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


def get_number_of_occurrences_in_doc(documents):
    word_occurrences = dict()
    for words in documents:
        unique_words = set(words)
        map(lambda w: add_one(word_occurrences, w), unique_words)

    # sort
    return collections.OrderedDict(sorted(word_occurrences.items()))


def add_one(dictionary, word):
    if word in dictionary:
        dictionary[word] += 1
    else:
        dictionary[word] = 1


def get_idf(total_documents, word_occurrences):
    idf_dict = dict()
    for key, value in word_occurrences.iteritems():
        inverse_freq = float(total_documents) / float(1 + value)
        idf_dict[key] = math.log(inverse_freq)

    values = idf_dict.values()
    return np.diag(values).astype(np.longdouble)


def get_empty_tf_matrix(rows, words):
    return sps.lil_matrix((rows, len(words)), dtype=np.longdouble)


def get_tf(total_documents, documents, words):
    matrix = get_empty_tf_matrix(total_documents, words)
    for row, doc in enumerate(documents):
        for word in doc:
            col = words.index(word)
            matrix[row, col] += 1
    return matrix


if __name__ == '__main__':
    docs = read_documents()
    occurrences = get_number_of_occurrences_in_doc(docs)
    idf = get_idf(TOTAL_NUMBER_OF_DOCUMENTS, occurrences)

    indexed_words = occurrences.keys()
    tf = get_tf(TOTAL_NUMBER_OF_DOCUMENTS, docs, indexed_words)

    tf_idf = np.dot(tf.toarray(), idf)
    print tf_idf