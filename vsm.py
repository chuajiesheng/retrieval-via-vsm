import math
import collections
import numpy as np
import scipy.sparse as sps
from itertools import islice


TOTAL_NUMBER_OF_DOCUMENTS = 2500
TOTAL_NUMBER_OF_QUERIES = 3
TOP_K = 10


def read_documents():
    documents = []
    for i in range(1, TOTAL_NUMBER_OF_DOCUMENTS + 1):
        document_name = 'documents/document%s.txt' % i
        words_in_doc = list_of_words(document_name)
        documents.append(words_in_doc)
    return documents


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
            if word not in words:
                continue

            col = words.index(word)
            matrix[row, col] += 1
            
    return matrix


def read_queries():
    documents = []
    for i in range(1, TOTAL_NUMBER_OF_QUERIES + 1):
        document_name = 'queries/query%s.txt' % i
        words_in_doc = list_of_words(document_name)
        documents.append(words_in_doc)
    return documents


def get_cosine_similarity(query_documents, documents):
    # row = queries
    # column = documents
    similarity = sps.lil_matrix((TOTAL_NUMBER_OF_QUERIES, TOTAL_NUMBER_OF_DOCUMENTS), dtype=np.longdouble)

    for row in range(TOTAL_NUMBER_OF_QUERIES):
        for col in range(TOTAL_NUMBER_OF_DOCUMENTS):
            similarity[row, col] = cosine_sim(query_documents[row, ], documents[col, ])

    return similarity


def cosine_sim(u, v):
    return np.dot(u, v) / (math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v, v)))


def get_top_10(similarity):
    list = []
    for row in range(TOTAL_NUMBER_OF_QUERIES):
        for col in range(TOTAL_NUMBER_OF_DOCUMENTS):
            list.extend({similarity[row, col]: col})

        sorted_similarity = collections.OrderedDict(sorted(list))
        print 'query', row
        print list(islice(sorted_similarity.iteritems(), TOP_K))


if __name__ == '__main__':
    docs = read_documents()
    occurrences = get_number_of_occurrences_in_doc(docs)
    idf = get_idf(TOTAL_NUMBER_OF_DOCUMENTS, occurrences)

    indexed_words = occurrences.keys()
    tf = get_tf(TOTAL_NUMBER_OF_DOCUMENTS, docs, indexed_words)

    tf_idf = np.dot(tf.toarray(), idf)

    queries = read_queries()
    query_occurrences = get_number_of_occurrences_in_doc(queries)
    query_tf = get_tf(TOTAL_NUMBER_OF_DOCUMENTS, queries, indexed_words)

    print 'query vector'
    print query_tf

    query_tf_idf = np.dot(query_tf.toarray(), idf)

    cosine_similarity = get_cosine_similarity(query_tf_idf, tf_idf)
    get_top_10(cosine_similarity)