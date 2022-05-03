import numpy as np
from numba import jit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from src.ndcg import ndcg_score


def print_evaluation(sparse_test, all_multinomials, method='auc', k='', _print=True):
    """
    Evaluates the model (resulted multinomials) on the test data, based on the method selected.
    Prints evaluation metric (averaged across users).

    :param sparse_test: CSR matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param method: string, can be 'logp' for log probability, 'recall',
    'precision', 'npk' 'ndcg', 'auc', 'prc_auc', 'micro_precision'
    :param k: the k from recall@k, various version of precision@k or nDCG@k.
    If method is logP, 'auc', 'prc_auc' then this does nothing.
    :param _print: whether the information is printed
    :return: method, score, and standard deviation if applicable
    """

    per_event = evaluate(sparse_test, all_multinomials, method, k)

    if k == '':
        _method = method
    elif k:
        _method = method + '@' + str(k)
    else:
        _method = method + '@' + 'M'

    if _print:
        print('%s: %.5f' % (_method, per_event))
    return _method, per_event


def evaluate(sparse_test, all_multinomials, method, k):
    """
    Evaluates the model (resulted multinomials) on the test data, based on the method selected.

    :param test_data: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param method: string, can be 'logp' for log probability, 'recall',
    'precision', 'npk' 'ndcg', 'auc', 'prc_auc', 'micro_precision'
    :param k: the k from recall@k, various version of precision@k or nDCG@k.
    If method is logP, 'auc', 'prc_auc' then this does nothing.
    :return: score, and standard deviation if applicable
    """

#     if method.lower() == 'logp':
#         test_points = np.repeat(test_data[:, :-1], test_data[:, -1].astype(int), axis=0).astype(int)
#         test_probs = all_multinomials[list(test_points.T)]
#         return np.mean(np.log(test_probs))

#     el
    if method.lower() == 'recall':
        return recall_at_top_k(sparse_test, all_multinomials, k)

    elif method.lower() == 'recall_unweighted':
        return unweighted_recall_at_top_k(sparse_test, all_multinomials, k)

    elif method.lower() == 'precision':
        return precision_at_top_k(sparse_test, all_multinomials, k)

    elif method.lower() == 'npk':
        return npk(sparse_test, all_multinomials, k)

    elif method.lower() == 'ndcg':
        return avg_ndcg(sparse_test, all_multinomials, k)
    elif method.lower() == 'user_ndcg':
        return avg_ndcg_users(sparse_test, all_multinomials, k)
    elif method.lower() == 'auc':
        return average_auc(sparse_test, all_multinomials)

    # elif method.lower() == 'micro_precision':
    #     return mirco_precision_at_top_k(sparse_test, all_multinomials, k)

    # elif method.lower() == 'auc_prc':
    #     return average_precision_recall_auc(sparse_test, all_multinomials)

    else:
        print('I do not know this evaluation method')


@jit(nogil=True)
def rates_to_exp_order(rates, argsort, exp_order, M):
    prev_score = 0
    prev_idx = 0
    prev_val = rates[argsort[0]]
    for i in range(1, M):
        if prev_val == rates[argsort[i]]:
            continue

        tmp = 0
        for j in range(prev_idx, i):
            exp_order[argsort[j]] = prev_score + 1
            tmp += 1

        prev_score += tmp
        prev_val = rates[argsort[i]]
        prev_idx = i

    # For the last equalities
    for j in range(prev_idx, i + 1):
        exp_order[argsort[j]] = prev_score + 1


@jit(nogil=True)
def rates_mat_to_exp_order(rates, argsort, exp_order, N, M):
    for i in range(N):
        rates_to_exp_order(rates[i], argsort[i], exp_order[i], M)


# @jit(cache=True)
@jit
def fix_exp_order(rates, exp_order, k, N):
    for i in range(N):
        mask = np.where(exp_order[i] <= k)[0]
        if len(mask) <= k:
            exp_order[i] = 0
            exp_order[i, mask] = 1
        else:
            max_val = np.max(exp_order[i, mask])
            max_val_mask = np.where(exp_order[i] == max_val)[0]
            exp_order[i] = 0
            exp_order[i, mask] = 1
            exp_order[i, max_val_mask] = (k - max_val + 1) / max_val_mask.shape[0]

def fix_exp_order_diff_k(rates, exp_order, k, N):
    # k is array of the same length of N
    for i in range(N):
        mask = np.where(exp_order[i] <= k[i])[0]
        if len(mask) <= k[i]:
            exp_order[i] = 0
            exp_order[i, mask] = 1
        else:
            max_val = np.max(exp_order[i, mask])
            max_val_mask = np.where(exp_order[i] == max_val)[0]
            exp_order[i] = 0
            exp_order[i, mask] = 1
            exp_order[i, max_val_mask] = (k[i] - max_val + 1) / max_val_mask.shape[0]

@jit
def recall_at_top_k(test_counts, scores, k):
    """
    Compute recall at k, Kotzias's version with frequency as weights

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    argsort = np.argsort(-scores, axis=1)
    exp_order = np.zeros(scores.shape)

    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    recall_in = test_counts.multiply(exp_order)
    u_recall = recall_in.sum(axis=1) / test_counts.sum(axis=1)  # gives runtime warning, but its ok. NaN are handled.
    u_recall = u_recall[~np.isnan(u_recall)]  # nan's do not count.
    return np.mean(u_recall)

def recall_at_top_k_users(test_counts, scores, k):
    """
    Compute recall at k, Kotzias's version with frequency as weights

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    argsort = np.argsort(-scores, axis=1)
    exp_order = np.zeros(scores.shape)

    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    recall_in = test_counts.multiply(exp_order)
    u_recall = recall_in.sum(axis=1) / test_counts.sum(axis=1)  # gives runtime warning, but its ok. NaN are handled.
    u_recall = u_recall[~np.isnan(u_recall)]  # nan's do not count.
    return u_recall

@jit
def unweighted_recall_at_top_k(test_counts, scores, k):
    """
    Compute common version of recall at k, without frequency as weights

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    argsort = np.argsort(-scores, axis=1)
    exp_order = np.zeros(scores.shape)

    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    recall_in = test_counts.multiply(exp_order)

    unweighted_test_counts = test_counts.astype(bool).astype(int).astype(float)
    unweighted_recall_in = recall_in.astype(bool).astype(int).astype(float)
    u_recall = unweighted_recall_in.sum(axis=1) / unweighted_test_counts.sum(
        axis=1)  # gives runtime warning, but its ok. NaN are handled.
    u_recall = u_recall[~np.isnan(u_recall)]  # nan's do not count.
    return np.mean(u_recall)


@jit
def precision_at_top_k(test_counts, scores, k):
    """
    Compute precision at k.

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """

    # obtain index order by high to low scores
    argsort = np.argsort(-scores, axis=1)
    # initialize exp_order with all 0s
    exp_order = np.zeros(scores.shape)
    # update exp_order: rank of item by predicted score for the user
    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    # update exp_order: item is in top-k recommendations for the user
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    # numerator in precision computation
    test_counts = test_counts.astype(bool).astype(int).astype(float)
    precision_in = test_counts.multiply(exp_order)
    u_precision = precision_in.sum(axis=1) / k
    # u_precision = u_precision[~np.isnan(u_precision)]
    return np.mean(u_precision)

@jit
def precision_at_top_k_users(test_counts, scores, k):
    """
    Compute precision at k.

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """

    # obtain index order by high to low scores
    argsort = np.argsort(-scores, axis=1)
    # initialize exp_order with all 0s
    exp_order = np.zeros(scores.shape)
    # update exp_order: rank of item by predicted score for the user
    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    # update exp_order: item is in top-k recommendations for the user
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    # numerator in precision computation
    test_counts = test_counts.astype(bool).astype(int).astype(float)
    precision_in = test_counts.multiply(exp_order)
    u_precision = precision_in.sum(axis=1) / k
    # u_precision = u_precision[~np.isnan(u_precision)]
    return u_precision


@jit
def npk(test_counts, scores, k):
    """
    Compute the normalized precision at k. >-< some error

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from normlized precision@k.
    :return: average score across users and the standard deviation
    """
    # to boolean
    test_counts = test_counts.astype(bool).astype(int).astype(float)
    # compute the numerator: minimum of k or count
    count_u = test_counts.astype(bool).astype(int).astype(float).sum(axis=1)
    count_u[count_u > k] = k

    # obtain index order by high to low scores
    argsort = np.argsort(-scores, axis=1)
    # initialize exp_order with all 0s
    exp_order = np.zeros(scores.shape)
    # update exp_order: rank of item by predicted score for the user
    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    # update exp_order: item is in top-k recommendations for the user
    fix_exp_order_diff_k(scores, exp_order, count_u, scores.shape[0])
    # numerator in precision computation
    precision_in = test_counts.multiply(exp_order)
    # for each user, obtain a value
    u_precision = precision_in.sum(axis=1) / count_u  # gives runtime warning, but its ok. NaN are handled.
    u_precision = u_precision[~np.isnan(u_precision)]  # nan's do not count.
    return np.mean(u_precision)


@jit
def avg_ndcg(test_data, all_multinomials, k):
    """
    Compute nDCG.

    :param test_data: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    dense_test = test_data.todense()
    scores = []
    for i, c in enumerate(dense_test):
        score = ndcg_score(c, [all_multinomials[i]], k=k, ignore_ties=False)
        scores.append(score)
    return np.mean(scores)

def avg_ndcg_users(test_data, all_multinomials, k):
    """
    Compute nDCG.

    :param test_data: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    dense_test = test_data.todense()
    scores = []
    for i, c in enumerate(dense_test):
        score = ndcg_score(c, [all_multinomials[i]], k=k, ignore_ties=False)
        scores.append(score)
    return scores


def average_auc(test_data, all_multinomials):
    """
    Compute AUC of ROC curve.

    :param test_data: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :return: average score across users and the standard deviation
    """
    dense_test = test_data.todense()
    scores = []
    for i, c in enumerate(dense_test):
        score = _auc(c.A1, all_multinomials[i])
        if score == score:
            scores.append(score)
    return np.mean(scores)


def _auc(y_true, y_scores):
    # binarization of integer array of y_true
    try:
        score = roc_auc_score(y_true.astype(bool).astype(int), y_scores)
        return score
    except ValueError:
        # if consuming of all/no items by a user, ignore such cases
        # Only one class present in y_true.
        # ROC AUC score is not defined in that case.
        return np.nan

# @jit
# def mirco_precision_at_top_k(test_data, all_multinomials, k):
#     """
#     Compute the micro precision at top k.
#
#     :param test_data: sparse matrix of the test data
#     :param all_multinomials: matrix of probability estimates of each user, item
#     :param k: the top k
#     :return: precision
#     """
#     test_data = np.asarray(test_data.todense())
#     all_multinomials = np.asarray(all_multinomials)
#     if k:
#         # selection for top k value
#         top_k_mask = all_multinomials.flatten().argsort()[-k:][::-1]
#         # effective selection only if score predicted >0
#         effective_mask = all_multinomials.flatten()[top_k_mask] > 0
#         effective_top_k = sum(effective_mask)
#         precision_in = sum(test_data.flatten()[top_k_mask][effective_mask] > 0)
#         precision = precision_in / effective_top_k
#     else:
#         # predicted true only if score predicted >0
#         effective_mask = all_multinomials.flatten() > 0
#         predicted_true = sum(effective_mask)
#         # True positive
#         precision_in = sum(test_data.flatten()[effective_mask] > 0)
#         precision = precision_in / predicted_true
#     return precision


@jit
def average_precision_recall_auc(sparse_test, all_multinomials):
    """
    Compute AUC of precision-recall curve.

    :param all_multinomials: matrix of probability estimates of each user, item
    :param test_data: COO matrix with test data.
    :return: average score across users.
    """
    dense_test = sparse_test.todense()
    scores = []
    for i, c in enumerate(dense_test):
        score = _precision_recall_auc(c.A1, all_multinomials[i])
        if score == score:  # nan cases: no plot, area is 0
            scores.append(score)
    return np.mean(scores)


@jit(nogil=True)
def _precision_recall_auc(y_true, y_scores):
    # binarization of integer array of y_true
    precision, recall, thresholds = precision_recall_curve(y_true.astype(bool).astype(int), y_scores)
    return auc(recall, precision)
