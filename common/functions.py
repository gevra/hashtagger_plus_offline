import numpy as np
import numpy.matlib as matlib
import ast
import math
import gc
import random
from itertools import combinations, chain, islice
from scipy.stats import kurtosis, skew
from collections import defaultdict
from datetime import date, datetime


def lower_keys(d, rename_keys_dict=None):
    if isinstance(d, list):
        return [lower_keys(dd, rename_keys_dict=rename_keys_dict) for dd in d]
    elif isinstance(d, dict):
        if rename_keys_dict is not None:
            for (old_key, new_key) in rename_keys_dict.items():  # use iteritems() in python2
                if old_key in d:
                    d[new_key] = d.pop(old_key)
        return dict((k.lower(), lower_keys(v, rename_keys_dict=rename_keys_dict)) for k, v in d.items())
    else:
        return d


def powerset(iterable, max_length=None):
    # from https://docs.python.org/2/library/itertools.html
    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if max_length is None:
        max_length = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(1, max_length+1))


def columns2list(df, columns):
    for c in columns:
        if not isinstance(df.iloc[-1][c], list):
            if 'datetime' in df.iloc[-1][c]:
                df[c] = df[c].map(incomplete_dt_list_converter)
            else:
                df[c] = df[c].map(incomplete_list_converter)
        # print(type(df.iloc[-1][c]))
    return df


def product_func(row):
    return row[0] * row[1]


def hmean(row):
    return 2 * row[0] * row[1] / (row[0] + row[1])


def incomplete_list_converter(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    else:
        return np.nan


def incomplete_dt_list_converter(x):
    if isinstance(x, str):
        return eval(x)
    else:
        return np.nan


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]


def count_stats(counts):
    # used in ./feature_extraction_functions.get_hashtag_cooccurrence_features()
    # counts is a list of integers starting from 0
    extended_list = [item for sublist in [c*[i] for i, c in enumerate(counts)] for item in sublist]
    avg = np.sum([i*c for i, c in enumerate(counts)]) * 1.0 / np.sum(counts)
    if avg != np.average(extended_list):
        print("smth went wrong with the average computation")
    median = int(np.median(extended_list))
    maxi = len(counts) - 1
    mini = min([i for i, c in enumerate(counts) if c > 0])
    std = np.std(extended_list)
    kurt = kurtosis(extended_list)
    skw = skew(extended_list)
    most_freq = np.argmax(counts)  # Only the first occurrence is returned.
    alone = counts[1] * 1.0 / np.sum(counts)

    return mini, maxi, avg, median, std, kurt, skw, most_freq, alone


def list2chunks(l, n=None, size=None):
    """Divide a list `l` in `n` chunks
    OR
    yield successive given-size chunks from l."""
    if (n is not None) and (size is not None):
        raise Exception("only the size or the number of chunks must be given, not both!")
    elif n is not None:
        l_c = iter(l)
        while 1:
            x = tuple(islice(l_c, n))
            if not x:
                return
            yield x
    elif size is not None:
        for i in range(0, len(l), size):
            yield l[i:i + size]
    else:
        raise Exception("either the size or the number of chunks must be given!")


def extract_floats_from_str(s):
    # as seen on https://stackoverflow.com/a/4289415/2262424
    list_of_floats = []
    for t in s.strip().split():
        try:
            list_of_floats.append(float(t))
        except ValueError:
            pass
    return list_of_floats


def match_all_in_list2df_col(p, list_of_tags, field_name):
    for t in list_of_tags:
        if t not in p[field_name]:
            return False
    return True


def match_any_in_list2df_col(p, list_of_tags, field_name):
    for t in list_of_tags:
        if t in p[field_name]:
            return True
    return False


def normalize_dict(d, normalize_to=1):
    """
    normalizes a dictionary values to a given number (default: 1)
    :param normalize_to: the desired sum of dictionary values
    :param d: dictionary with numerical values
    :return:
    """
    sum_of_scores = sum(d.values())
    return dict([(t, score * normalize_to / sum_of_scores) for (t, score) in d.items()])


def merge_lists_with_common_items(list_of_lists):
    # found on https://stackoverflow.com/a/4842897/2262424
    out = []
    while len(list_of_lists) > 0:
        first, *rest = list_of_lists
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        out.append(list(first))
        list_of_lists = rest
    return out


def find_curve_elbow_idx_based_on_max_dist(curve):
    # found at https://stackoverflow.com/a/37121355/2262424
    n_points = len(curve)
    coords = np.vstack((list(range(n_points)), curve)).T
    # coords = np.array([range(n_points), curve])
    first_point = coords[0]
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = coords - first_point
    scalar_product = np.sum(vec_from_first * matlib.repmat(line_vec_norm, n_points, 1), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    return np.argmax(dist_to_line)


def find_curve_elbow_2nd_derivative(curve):
    # NOTE! this doesn't work if the curve is not monotonic and has some fluctuations!
    # NOTE! the min or max will depend on convex/concave shape of the curve!
    second_derivative = []
    for i in range(1, len(curve) - 1):
        second_derivative.append(curve[i + 1] + curve[i - 1] - 2 * curve[i])
    second_derivative = [0] + second_derivative + [0]
    return second_derivative.index(min(second_derivative))


def sliding_window(seq, window_size=2):
    """
    https://stackoverflow.com/a/6822773/2262424
    Returns a sliding window (of width window_size) over data from the iterable
    s -> (s0,s1,...s[window_size-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def count_digits(integer):
    # from https://stackoverflow.com/a/2189827/2262424
    # though keep in mind that the int() may give issues because of the floating point output of log10()
    # using round() may fix the problem and it's not much more expensive https://stackoverflow.com/a/44920965/2262424
    if integer > 0:
        # n_digits = int(math.log10(integer)) + 1
        n_digits = round(math.log10(integer)) + 1
    elif integer == 0:
        n_digits = 1
    else:
        # n_digits = int(math.log10(-integer)) + 1  # +2 if you count the '-'
        n_digits = round(math.log10(-integer)) + 1  # +2 if you count the '-'
    return n_digits


def queryset_iterator(queryset, chunksize=10000):
    """
    Iterate over a Django Queryset ordered by the primary key

    This method loads a maximum of chunksize (default: 10000) rows in it's
    memory at the same time while django normally would load all rows in it's
    memory. Using the iterator() method only causes it to not preload all the
    classes.

    Note that the implementation of the iterator does not support ordered query sets.

    usage:
        my_queryset = queryset_iterator(MyItem.objects.all())
        for item in my_queryset:
            item.do_something()

    from https://djangosnippets.org/snippets/1949/
    you need this because https://stackoverflow.com/a/4222432/2262424
    """
    pk = 0
    last_pk = queryset.order_by('-pk')[0].pk
    queryset = queryset.order_by('pk')
    while pk < last_pk:
        for row in queryset.filter(pk__gt=pk)[:chunksize]:
            pk = row.pk
            yield row
        gc.collect()


def ordered_sample_without_replacement(seq, sample_size):
    """
    Returns a random sample from the list while preserving the item order in the list
    O(N) solution from https://stackoverflow.com/a/6482925/2262424
    :param seq: list of items
    :param sample_size: sample size
    :return:
    """
    total_items = len(seq)
    if not 0 <= sample_size <= total_items:
        raise ValueError('Required that 0 <= sample_size <= population_size')

    picks_remaining = sample_size
    for seen_items, element in enumerate(seq):
        items_remaining = total_items - seen_items
        prob = picks_remaining / items_remaining
        if random.random() < prob:
            yield element
            picks_remaining -= 1


def reverse_dict_of_lists(d):
    """
    Inverts a lookup table with lists.
    for input d = {"abc": [1, 2, 3], "cde": [3, 5, 7], "efg": [4, 2, 1, 7]}
    outputs {1: ['abc', 'efg'], 2: ['abc', 'efg'], 3: ['cde', 'abc'], 4: ['efg'], 5: ['cde'], 7: ['cde', 'efg']}

    from https://codereview.stackexchange.com/a/183173/181366
    :param d: dictionary
    :return: dictionary
    """
    reversed_dict = defaultdict(list)
    for key, values in d.items():
        for value in values:
            reversed_dict[value].append(key)
    return reversed_dict


def dict_cosine_similarity(reference_dict, target_dict):
    scalar_product = 0
    for k in reference_dict.keys():
        if k in target_dict.keys():
            scalar_product += reference_dict[k] * target_dict[k]

    reference_dict_vector_length = np.sqrt(sum(v ** 2 for v in reference_dict.values()))
    target_dict_vector_length = np.sqrt(sum(v ** 2 for v in target_dict.values()))

    if reference_dict_vector_length * target_dict_vector_length == 0:
        return 0
    else:
        cosine_similarity = scalar_product / (reference_dict_vector_length * target_dict_vector_length)
        return cosine_similarity


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code
    from https://stackoverflow.com/a/22238613/2262424"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))
