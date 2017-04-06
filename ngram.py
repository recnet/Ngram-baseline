from numpy import dot
from math import sqrt
import csv_reader


def n_gram(text, n):
    """Returns a generator for all n-grams of the text"""
    grams = (zip(*[text[i:] for i in range(n)]))
    return grams


def all_grams(text, n=3):
    """Returns a list of all n-grams up to n based on text, by default n is 3"""
    grams = []
    for i in range(n):
        grams.extend(n_gram(text, i+1))
    return grams


def build_index_table(words):
    """
    :param words: list of words
    :return dictionary giving each word a unique index
    """
    grams = set(all_grams(words))
    table = {}
    for index, gram in enumerate(grams):
        table[gram] = index
    return table


def build_title_vector(table, words):
    """
    :param table: dictionary
    :param words: list of words to build the n-grams from
    :return: n-gram vector for the words
    """
    grams = all_grams(words)
    vector = [0]*len(table)
    for gram in grams:
        index = table[gram]
        vector[index] += 1
    return vector


def cosine_similarity(category_vector, title_vector):
    """
    :param category_vector: category vector
    :param title_vector: title vector
    :return: cosine distance between the two given vectors
    """
    inner = dot(category_vector, title_vector)
    dist_category = sqrt(sum(map(lambda x: x**2, category_vector)))
    dist_title = sqrt(sum(map(lambda x: x**2, title_vector)))

    cosine = inner / (dist_category*dist_title)
    return cosine


def all_cosines(category_vector_table, title_vector):
    """
    :param category_vector_table: dictionary mapping each user to their category vector
    :param title_vector: vector to compare against
    :return: list of tuples of user and cosine between the title_vector and that users category vector
    """
    results = [None]*len(category_vector_table.keys())
    for index, user in enumerate(category_vector_table.keys()):
        cosine = cosine_similarity(category_vector_table[user], title_vector)
        results[index] = (cosine, user)
    return results


def classify(category_vector_table, title_vector, predicate=lambda x: [max(x)]):
    """
    :param category_vector_table: dictionary mapping each user to their category vector
    :param title_vector: vector to compare against
    :param predicate: function that takes a list of cosine, user tuples return which ones to recommend
    :return: predicate function applied with the list of cosine,user tuples
    """
    cosines = all_cosines(category_vector_table,title_vector)
    return predicate(cosines)


def user_titles_table(raw_titles, raw_users):
    """
    :param raw_titles: list of titles from csv_reader
    :param raw_users: list of users from csv_reader
    :return: dictionary from users to a single long title (all their titles concatenated)
    """
    table = {}

    for title, user in zip(raw_titles, raw_users):
        if user not in table:
            table[user] = []
        table[user].append(title)

    for key in table.keys():
        table[key] = " ".join(table[key])

    return table


def make_category_table(grams_index_table, user_title_table):
    """
    :param grams_index_table: dictionary mapping gram to index
    :param user_title_table: dictionary mapping user to a single long title (all their titles concatenated)
    :return: dictionary mapping user to their category vector
    """
    table = {}
    for user in user_title_table.keys():
        table[user] = build_title_vector(grams_index_table, user_title_table[user].split())
    return table


def read(file_path):
    "titles, users"
    return csv_reader.CsvReader().get_data(file_path)


def print_stats(true_positive, true_negative, false_negative, false_positive, total):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * ((precision*recall) / (precision+recall))
    print("Correct guess {0}% of the time (precision {1} )".format(100*precision, precision))
    print("recall is {0} ".format(recall))
    print("F1 score {0} ".format(f1_score))

if __name__ == "__main__":
    path = "data/training_data_top_n_single.csv"
    titles, users = read(path)
    path = "data/validation_data_top_n_single.csv"
    val_titles, val_users = read(path)

    user2full_title = user_titles_table(titles, users)
    user2full_title_val = user_titles_table(val_titles, val_users)
    print("number of users {0}".format(len(user2full_title_val)))

    # Don't need these anymore
    del titles
    del users

    # Big index table needs all grams (validation and training)
    words = []
    for user in user2full_title.keys():
        words.extend(user2full_title[user].split())

    for user in user2full_title_val.keys():
        words.extend(user2full_title_val[user].split())

    all_grams_index_table = build_index_table(words)

    del words

    # categories are based on the training data
    categories = make_category_table(all_grams_index_table, user2full_title)

    del user2full_title

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    count = 0

    def top2(x):
        x1 = max(x)
        x2 = max(set(x)-set(x1))
        return [x1, x2]

    for title, user in zip(val_titles, val_users):
        title_vector = build_title_vector(all_grams_index_table, title.split())
        predictions = classify(categories, title_vector, predicate=top2)
        predictions = list(map(lambda x: x[1], predictions))

        for prediction in predictions:
            if prediction == user:
                # should recommend, did recommend
                true_positive += 1
            else:
                # should not recommend, did recommend
                false_positive += 1
        for not_prediction in set(val_users) - set(predictions):
            if not_prediction == user:
                # should recommend, did not recommend
                false_negative += 1
            else:
                # should not recommend, did not recommend
                true_negative += 1

        count += 1
        if count % 50 == 0:
            print("finished {0} iterations out of a total of {1}".format(count, len(val_titles)))
            print_stats(true_positive, true_negative, false_negative, false_positive, count)

    print_stats(true_positive, true_negative, false_negative, false_positive, count)
    print("True positives {0} ".format(true_positive))
    print("True negative {0} ".format(true_negative))
    print("false negative {0} ".format(false_negative))
    print("false positives {0} ".format(false_positive))

