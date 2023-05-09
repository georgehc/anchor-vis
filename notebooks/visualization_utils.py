import numpy as np
import os
from sklearn.model_selection import train_test_split


def get_experiment_data(dataset, output_dir, seed=0, direction=None, train_size=0.25,
                        random_state=1452512785):
    base_dir = os.path.join(output_dir,
                            'cached_embeddings/%s_exp%d' % (dataset, seed))
    test_embeddings = \
        np.loadtxt(os.path.join(base_dir, 'test_embeddings.txt'))
    test_y = np.loadtxt(os.path.join(base_dir, 'test_y.txt'))
    feature_names = np.load(os.path.join(base_dir, 'feature_names.npy'), allow_pickle=True)
    unique_train_times = np.loadtxt(os.path.join(base_dir, 'train_unique_event_times.txt'))
    predicted_surv_curves = np.loadtxt(os.path.join(base_dir, 'test_pred_surv.txt'))

    if dataset == 'survival-mnist':
        test_X = np.load(os.path.join(base_dir, 'test_X.npy'), allow_pickle=True)
        test_digits = np.loadtxt(os.path.join(base_dir, 'test_digits.txt'))
        emb_direction, emb_vis, raw_direction, raw_vis, label_direction, \
                label_vis, _, predicted_surv_vis, digit_direction, digit_vis \
            = train_test_split(test_embeddings, test_X, test_y,
                               predicted_surv_curves, test_digits,
                               train_size=train_size,
                               random_state=random_state)
        return emb_direction, emb_vis, raw_direction, raw_vis, \
            label_direction, label_vis, feature_names, unique_train_times, \
            predicted_surv_vis, digit_direction, digit_vis
    else:
        test_X = np.loadtxt(os.path.join(base_dir, 'test_X.txt'))
        emb_direction, emb_vis, raw_direction, raw_vis, label_direction, \
                label_vis, _, predicted_surv_vis \
            = train_test_split(test_embeddings, test_X, test_y,
                               predicted_surv_curves, train_size=train_size,
                               random_state=random_state)
        return emb_direction, emb_vis, raw_direction, raw_vis, \
            label_direction, label_vis, feature_names, unique_train_times, \
            predicted_surv_vis


def l2_normalize_rows(X):
    return X / np.sqrt(np.einsum('...i,...i', X, X))[..., np.newaxis]


# https://www.geeksforgeeks.org/python-program-to-find-longest-common-prefix-using-sorting/
def longest_common_prefix(a):      
    size = len(a)

    # if size is 0, return empty string 
    if (size == 0):
        return ""

    if (size == 1):
        return a[0]

    # sort the array of strings 
    a.sort()

    # find the minimum length from 
    # first and last string 
    end = min(len(a[0]), len(a[size - 1]))

    # find the common prefix between 
    # the first and last string 
    i = 0
    while (i < end and a[0][i] == a[size - 1][i]):
        i += 1

    pre = a[0][0: i]
    return pre


def compute_median_survival_times(surv_functions, time_grid):
    single_surv_function = (len(surv_functions.shape) == 1)
    if single_surv_function:
        surv_functions = np.array([surv_functions])
    median_surv_time_estimates = np.zeros(surv_functions.shape[0])
    for idx, row in enumerate(1*(surv_functions <= 0.5)):
        if np.any(row):
            median_surv_time_estimates[idx] = time_grid[np.where(row)[0][0]]
        else:  # survival curve never crosses 1/2
            median_surv_time_estimates[idx] = np.inf
    if single_surv_function:
        return median_surv_time_estimates[0]
    else:
        return median_surv_time_estimates
