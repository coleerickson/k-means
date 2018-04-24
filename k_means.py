from collections import Counter, defaultdict
from functools import reduce
from itertools import groupby
from parse_arff import Database
from pprint import pformat, pprint
from random import sample

def squared_distance(x, y):
    ''' Returns the square of the Euclidean distance between vectors `x` and
    `y`, where `x` and `y` are represented as lists. '''
    return sum((xi - yi) ** 2 for (xi, yi) in zip(x, y))

def vector_sum(x, y):
    ''' Pointwise sum of vectors `x` and `y`, as lists. '''
    return [xi + yi for xi, yi in zip(x, y)]

def vector_scalar_div(x, factor):
    return [xi / factor for xi in x]

def most_common_element(counter: Counter):
    return counter.most_common(1)[0][0]

def format_vec(v):
    return ', '.join('{0:10.4f}'.format(vi) for vi in v)

def compute_clusters(examples, ref_vectors):
    clusters = defaultdict(list)

    # Iterate over every example and assign it to a cluster.
    for i, example in enumerate(examples):
        features = example[:-1]
        # The "cluster" is the index of the nearest reference vector.
        nearest_ref_vector_index = min(enumerate(ref_vectors), key=lambda index_ref_vector_pair: squared_distance(index_ref_vector_pair[1], features))[0]
        clusters[nearest_ref_vector_index].append(example)
    return clusters

class KMeans:
    def __init__(self, k, database):
        self.database = database
        self.k = k
        self.ref_vectors = self._learn_reference_vectors()

    def _learn_reference_vectors(self):
        examples = self.database.data
        num_features = len(examples[0]) - 1

        # The k reference vectors. Initially, these are just randomly
        # selected examples from the training data.
        ref_vectors = sample(examples, self.k)
        ref_vectors = [ref_vector[:-1] for ref_vector in ref_vectors]

        iterations = 0
        while True:
            iterations += 1
            clusters = compute_clusters(examples, ref_vectors)

            has_converged = True
            for ref_vector_index, cluster_examples in clusters.items():
                old_ref_vector = ref_vectors[ref_vector_index]
                cluster_features = [cluster_example[:-1] for cluster_example in cluster_examples]
                cluster_summation = reduce(vector_sum, cluster_features, [0] * num_features)
                # pprint(cluster_summation)
                new_ref_vector = vector_scalar_div(cluster_summation, len(cluster_examples))
                # print('new ref vecotr is')
                # pprint(new_ref_vector)
                if new_ref_vector != old_ref_vector:
                    has_converged = False
                ref_vectors[ref_vector_index] = new_ref_vector
            if has_converged:
                break

        # "Name" the reference vectors with the most common class of the examples
        # belonging to the cluster.
        for ref_vector_index, cluster in clusters.items():
            ref_vector_name = most_common_element(Counter(example[-1] for example in cluster))
            ref_vectors[ref_vector_index].append(ref_vector_name)


        print('Finished after %d iterations' % iterations)
        return ref_vectors

    def get_clusters(self):
        return compute_clusters(self.database.data, self.ref_vectors)

    def compute_misclassifications(self):
        ''' Returns a dictionary which maps from the learned reference vectors to the number of
        misclassifications in its cluster. '''
        clusters = self.get_clusters()
        misclassification_counts = {}
        for ref_vector_index, cluster_examples in clusters.items():
            ref_vector = self.ref_vectors[ref_vector_index]
            num_misclassified = sum(int(ex[-1] != ref_vector[-1]) for ex in cluster_examples)
            misclassification_counts[ref_vector_index] = num_misclassified
        return misclassification_counts

    def __str__(self):
        attribute_names = self.database.ordered_attributes[:-1]
        attribute_name_format_string = '  '.join(['{:10s}'] * (len(attribute_names)))
        column_headers = attribute_name_format_string.format(*attribute_names)
        formatted_ref_vectors = '\n'.join([format_vec(ref_vector[:-1]) + \
            ',     with class: ' + \
            self.database.attributes[self.database.ordered_attributes[-1]][ref_vector[-1]]
            for ref_vector in self.ref_vectors])
        return 'Reference vectors:\n' + \
            column_headers + '\n' + \
            formatted_ref_vectors


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='K Means')

    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='.arff file name', required=True)
    parser.add_argument('-k', '--k', type=int, help='The number of means to compute', required=True)

    args = parser.parse_args()

    file_name = args.file
    k = args.k

    db = Database()
    db.read_data(file_name)

    k_means = KMeans(k, db)
    print()
    print(k_means)
    print()


    print('Misclassified instances:')
    misclassifications = k_means.compute_misclassifications()
    # pprint(misclassifications)
    clusters = k_means.get_clusters()
    summary_lines = []
    for ref_vector_index, cluster_examples in clusters.items():
        num_misclassifications = misclassifications[ref_vector_index]
        cluster_size = len(cluster_examples)
        ref_vector = k_means.ref_vectors[ref_vector_index]
        cluster_name = k_means.database.attributes[k_means.database.ordered_attributes[-1]][ref_vector[-1]]
        # cluster_name = k_means.ref_vectors[ref_vector_index][-1]

        summary_lines.append('%30s: %5d  /%5d   =  %0.3f%%' % (cluster_name, num_misclassifications, cluster_size, 100 * (num_misclassifications / cluster_size)))
    summary_lines.sort()
    print('\n'.join(summary_lines))
