import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.evaluation import print_evaluation
from src.array_util import data_to_sparse
from src.data_io import get_dataset, get_decay_dataset, get_fpmc_dataset
from src.io import load_pickle, save_pickle


def model_results(datasets, train_model, model_type, data_dir, results_dir, eval_lst, overwrite=False,
                  save_multinomials=True, save_model=False, is_eval=True, n_components=50, top_n=10, n_epoch=150,
                  regularization=0.001, n_neg=10, lr=0.01, alpha=0, new_items=False, filename=False, param=False,
                 on_val=False, return_multinomials=False):
    """ Method that trains and evaluates for a model type with all dataset.

    :param datasets: List of dataset names.
    :param train_model: Model function to be used.
    :param model_type: Name of the model.
    :param data_dir: Name of the directory the data will be retrieved from.
    :param results_dir: Name of the directory the results will be saved.
    :param n_components: Number of components in FPMC, NMF, LDA, HPF, WRMF.
    :param n_epoch: Number of epochs iterated.
    :param regularization: Degree of regularization for WRMF model and FPMC model.
    :param n_neg: Number of negative samples per batch.
    :param lr: The learning rate.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param save_model: Boolean, on whether to save the model parameters.
    :param alpha: Weight on purchases for WRMF model.

    :return: return DataFrame output
    """
    if not filename:
        filename = os.path.join(results_dir, '_'.join([model_type, 'eval_results.pkl']))

    if os.path.exists(filename) and not overwrite:
        output = load_pickle(filename)
        return output
    else:
        results = []

    for dataset in datasets:
        
        train, val, test = get_dataset(dataset, on_val=on_val)
        
        if isinstance(n_components, (dict,)):
            n_compo = n_components[dataset]
        else:
            n_compo = n_components
        if param:
            # For WRMF model
            n_compo = param[dataset]['n_component']
            alpha = param[dataset]['alpha']
            regularization = param[dataset]['lambda']
         
        if model_type in ['nmf_model', 'lda_model', 'wrmf_model', 'bpr_model']:
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite, save_multinomials,
                                           n_compo, is_eval)
        elif model_type in ['wrmf_model']:
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite, save_multinomials,
                                           n_compo, regularization, alpha, is_eval)
        elif model_type == 'mixture_model':
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite, save_multinomials)
            
        elif model_type == 'mixture_decay_model':
            train_decay, val_decay, test_decay = get_decay_dataset(dataset, decay=n_compo)
            all_multinomials = train_model(train_decay, val_decay, test_decay, results_dir, dataset, overwrite,
                                           save_multinomials, model_type=model_type)

        elif model_type == 'fpmc_model':
            input_data = get_fpmc_dataset(dataset)
            all_multinomials = train_model(input_data, results_dir, dataset, n_components, n_epoch, regularization, n_neg,
                                           lr, overwrite, save_multinomials, save_model)
        else:
           all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite,
                                           save_multinomials, is_eval)
 
        if new_items:
            prev_items = data_to_sparse(np.vstack((train, val))).todense()
            # exclude repeated items in the predictions
            all_multinomials[np.where(prev_items != 0)] = 0
            # exclude repeated items in the test set
            test_mat = data_to_sparse(test).todense()
            test_mat[np.where(prev_items != 0)] = 0
            sparse_test = csr_matrix(test_mat)
        else:
            sparse_test = data_to_sparse(test)
            
        if return_multinomials:
            return all_multinomials

        result = _results(sparse_test, all_multinomials, eval_lst, dataset)
        results.append(result)

    output = format_results(results)
    save_pickle(filename, output, False)

    return output


def _results(sparse_test, all_multinomials, eval_lst, dataset):  
    eval_result = []
    for _method, k_vals in eval_lst:
        for k in k_vals:
            method, value = print_evaluation(sparse_test, all_multinomials, _method, k)
            eval_result.append([method, value])
    return [dataset, eval_result]


def format_results(results):
    header = ['dataset'] + ordered_set([item[0] for sublist in results for item in sublist[1]])
    output = pd.DataFrame([[line[0]] + [pair[1] for pair in line[1]]
                           for line in results], columns=header).set_index('dataset')
    return output


def ordered_set(x):
    return sorted(set(x), key=x.index)
