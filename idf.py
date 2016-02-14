import math


def process_document():
    for line in open(document_name):
        words = process_line(line)
        map(lambda w: add_one(occurrences, w), words)


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

total_number_of_documents = 2500
occurrences = dict()

for i in range(1, total_number_of_documents + 1):
    document_name = 'documents/document%s.txt' % i
    process_document()

idf = dict()

for key, value in occurrences.iteritems():
    inverse_freq = float(total_number_of_documents) / float(1 + value)
    idf[key] = math.log(inverse_freq)

print idf

