import os
import numpy as np
import implicit
from lightfm import LightFM
from scipy.sparse import coo_matrix
from sklearn.decomposition import LatentDirichletAllocation, NMF

from src.mixture_functions import get_train_global, learn_individual_mixing_weights, \
    learn_global_mixture_weights
from src.fpmc_functions import FPMC, data_to_3_list
from src.mixture_functions import get_train
from src.all_evaluation import model_results
from src.io import load_pickle, save_pickle
from src.array_util import data_to_sparse


DATA_DIR, MODEL_DIR, PARAM_DIR, OUTPUT_DIR, datasets = (
     '/data/yueliu/Recommendation/data/last_basket_data',
     '/data/yueliu/Recommendation/model',
     '/data/yueliu/Recommendation/output/param',
     '/data/yueliu/Recommendation/output/result',
     ['MFP47K_last basket'])
     
eval_lst = [['recall', [10]], ['precision', [10,]], ['nDCG', [10]]]




def learn_mixing_weights(components, validation_data, num_proc=None):
    """Runs the Smoothed Mixture model on the number of components. Each component is an array of UID x PID
    (user x items). This runs in parallel for efficiency.

    :param components: List of matrices (CSR or full) -- all must have the same size.
    :param validation_data: COO matrix of validation data.
    :param num_proc: Number of processes to be used. If none, all the processors in the machine will be used.
    :return return the mixing weights for each user (or the entire population).
    """
    alpha = 1.001  # very small prior for global.
    global_mix_weights = learn_global_mixture_weights(alpha, components, validation_data)  # learn global mix weights.
    val_data = data_to_sparse(validation_data)
    # use global mixing weights as prior for individual ones.
    user_mix_weights = learn_individual_mixing_weights(global_mix_weights, components, val_data, num_proc)
    return user_mix_weights


def _user_multinomials(train, val, test, mix_weights):
    """
    Predicts the scores of users for items.
    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param mix_weights: the mixing weights for each user (or the entire population).
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    eval_train_matrix, eval_glb_matrix = get_train_global(train, val, test, is_eval=True)
    eval_components = [eval_train_matrix, eval_glb_matrix]
    user_multinomials = mix_multinomials(eval_components, mix_weights)
    return user_multinomials


def mix_multinomials(components, mixing_weights):
    """Returns the multinomial distribution for each user after mixing them.

    :param components: List of matrices (CSR or full) -- all must have the same size.
    :param mixing_weights: List of mixing weights (for each component). If mixing weights is an array, then it means
    there is one mixing weight per user. Otherwise, they are global.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    if mixing_weights.ndim == 2:
        return _user_individual_multinomial(components, mixing_weights)
    else:
        return _user_global_multinomials(components, mixing_weights)


def _user_global_multinomials(components, mixing_weights):
    """The mixing weights are the same for each user.
    :param components: List of components
    :param mixing_weights: List of mixing weights (one for each component)
    :return: return multinomials, which is a dense matrix of predicted probability
    """

    result = np.zeros(components[0].shape)
    for i, c in enumerate(components):
        result += mixing_weights[i] * c
    return np.array(result)


def _user_individual_multinomial(components, mixing_weights):
    """ The mixing weights are different for each user.
    :param components: List of components
    :param mixing_weights: List of mixing weights (for each component, and each user)
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    result = np.zeros(components[0].shape)
    for i, c in enumerate(components):
        c = np.array(c.todense())
        result += mixing_weights[:, i][:, np.newaxis] * c
    return result

def train_global_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                       is_eval=True):
    """
    Runs the Global experiment of the paper.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    filename = os.path.join(results_dir, 'global_model', dataset_name, 'user_multinomials.pkl')
    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        if is_eval:
            train = np.vstack((train, val))
        train_matrix = data_to_sparse(train).tocsr()
        all_multinomials = construct_multinomials(train_matrix, test)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials


def construct_multinomials(train_matrix, test):
    shape = data_to_sparse(test).shape
    item_weights = train_matrix.sum(axis=0)
    result = np.repeat(item_weights, shape[0], axis=0)
    result = np.asarray(result, dtype=np.float32)
    return result


def train_favourite_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                          is_eval=True):
    """
    Runs the Personal experiment of the paper.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    filename = os.path.join(results_dir, 'personal_model', dataset_name, 'user_multinomials.pkl')
    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        train_matrix = get_train(train, val, test, is_eval)
        all_multinomials = construct_multinomials_favourite(train_matrix)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials


def construct_multinomials_favourite(train_matrix):
    row_sums = train_matrix.sum(axis=1)
    new_matrix = train_matrix / row_sums
    return np.asarray(new_matrix)


def train_bpr_model(train, val, test, results_dir, dataset_name,
                    overwrite=False, save_multinomials=True, n_components=500,
                    is_eval=True, num_epochs=100, num_threads=4):
    """
    Runs the BPR-MF experiment with lightFM implementation. 

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data ndarray, to be converted to COO matrix.
    :param val: Validation data ndarray, to be converted to COO matrix.
    :param test: Test data ndarray, to be converted to COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param n_components: Number of components in NMF.
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
   
    filename = os.path.join(results_dir, 'bpr_model', dataset_name, 'user_multinomials.pkl')

    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename, False)
    else:
        train = data_to_sparse(train)
        val = data_to_sparse(val)
        if is_eval:
            train = train + val  # CSR <= COO + COO
        model = LightFM(no_components=n_components, loss='bpr', learning_rate=0.05)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)
        all_multinomials = construct_multinomials(model, train.shape, num_threads)

        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials


def construct_multinomials(model, shape, num_threads):
    result = np.zeros(shape, dtype=np.float32)
    n_items = shape[1]
    pid_array = np.array(np.arange(n_items), dtype=np.int32)
    for uid in range(shape[0]):
        uid_array = np.array([uid] * n_items, dtype=np.int32)
        predictions = model.predict(uid_array, pid_array, num_threads=num_threads)
        result[uid] += predictions
    return result




def train_lda_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                    n_components=50, is_eval=True):
    """
    Runs the LDA experiment with scikit-learn implementation. It evaluates on the test set.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param n_components: Number of components in LDA.
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :return: return multinomials, which is a dense matrix of predicted probability
    """

    filename = os.path.join(results_dir, 'lda_model', dataset_name, str(n_components), 'user_multinomials.pkl')

    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        model = LatentDirichletAllocation(n_components=n_components, random_state=0)
#         if is_eval:
#             test_matrix = get_test(train, val, test)
#         else:
#             test_matrix = get_val(train, val, test)
        train_matrix = get_train(train, val, test, is_eval)
        model.fit(train_matrix)
        all_multinomials = model.transform(train_matrix).dot(model.components_)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials


def train_nmf_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                    n_components=100, is_eval=True):
    """
    Runs the NMF experiment with scikit-learn implementation. It evaluates on the test set.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param n_components: Number of components in NMF.
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :return: return multinomials, which is a dense matrix of predicted probability
    """

    filename = os.path.join(results_dir, 'nmf_model', dataset_name, str(n_components), 'user_multinomials_.pkl')

    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        model = NMF(n_components=n_components, init='random', random_state=0)
#         if is_eval:
#             test_matrix = get_test(train, val, test)
#         else:
#             test_matrix = get_val(train, val, test)
        train_matrix = get_train(train, val, test, is_eval)
        model.fit(train_matrix)
        all_multinomials = model.transform(train_matrix).dot(model.components_)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials




def train_fpmc_model(input_data, results_dir, dataset_name, n_components, n_epoch=100, regular=0.001, n_neg=10,
                     lr=0.01, overwrite=False, save_multinomials=True, save_model=False):
    """
    Runs the FPMC experiment. Adapted from https://github.com/khesui/FPMC} to allow for variable-sized baskets.

    :param input_data: tr_data, te_data, test, user_set, item_set.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param n_components: Number of components in FPMC.
    :param n_epoch: Number of epochs iterated.
    :param regular: Degree of regularization.
    :param n_neg: Number of negative samples per batch.
    :param lr: The learning rate.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param save_model: Boolean, on whether to save the model parameters.
    :return: return multinomials, which is a dense matrix of predicted probability
    """

    filename = os.path.join(results_dir, 'fomc_model', dataset_name, str(n_components), 'user_multinomials_.pkl')
    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        all_multinomials, model = _train_fpmc_model(input_data, n_components, n_epoch, regular, n_neg, lr)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
        if save_model:
            save_model_param(model, results_dir, 'fomc_model', dataset_name)
    return all_multinomials


def _train_fpmc_model(input_data, n_components, n_epoch, regular, n_neg, lr):
    tr_data, te_data, test, user_set, item_set = input_data
    fpmc = FPMC(n_user=max(user_set) + 1, n_item=max(item_set) + 1, n_factor=n_components, learn_rate=lr,
                regular=regular)
    fpmc.user_set = user_set
    fpmc.item_set = item_set
    fpmc.init_model()
    tr_3_list = data_to_3_list(tr_data)
    for epoch in range(n_epoch):
        fpmc.learn_epoch(tr_3_list, n_neg)
    multinomials = fpmc.construct_multinomials(te_data)
    return multinomials, fpmc


def save_model_param(model, results_dir, model_type, dataset_name):
    for component, item in {'VUI': model.VUI, 'VIU': model.VIU, 'VLI': model.VLI, 'VIL': model.VIL}.items():
        filename = os.path.join(results_dir, model_type, dataset_name, component + '.pkl')
        save_pickle(filename, item, False)
        
        
def train_wrmf_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                    n_components=50, regularization=0.01, alpha=0, is_eval=True, top_n=False):
    """
    Runs the WRMF experiment with implicit implementation. It evaluates on the test set.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix of same shape as train.
    :param test: Test data COO matrix of same shape as train.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the dataset.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param n_components: Number of components in WRMF.
    :param regularization: L2 regularization.
    :param alpha: Weight on purchases.
    :param is_eval: Boolean, if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :param top_n: Number of predictions per user. Higher number requires more computational power. If set as False,
    all items are predicted for all users.

    :return: return multinomials, which is a dense matrix of predicted probability
    """
    filename = os.path.join(results_dir, 'wrmf_model', dataset_name, str(n_components), 'user_multinomials.pkl')
    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        if is_eval:
            train = np.vstack((train, val))
        train_matrix = data_to_sparse(train)
        if alpha >= 0:
            # For basic mapping schemes such as c_ui = 1+ alpha * r_ui
            train_matrix = alpha * train_matrix
            # May also takes the other mapping schemes such as c_ui = 1+ alpha*log(1+ r_ui/(10^-8))
        train_matrix.data += 1
            
        num_users, num_items = train_matrix.shape
        
        # If top_n != num_items (e.g. top 10 for tuning), nDCG and AUC cannot be evaluated accurately.
        if not top_n:
            top_n = num_items
            
        # initialize a model
        model = implicit.als.AlternatingLeastSquares(factors=n_components, regularization=regularization, 
                                                     iterations=3*n_components, calculate_training_loss=True)
        # model is trained with item_user_data
        model.fit(train_matrix.T)
        pred_item = np.array([]).astype(int)
        pred_user = np.array([]).astype(int)
        pred_prob = np.array([]).astype(float)
        for user in np.arange(num_users):
            # recommend items for a user
            recommendations = model.recommend(user, train_matrix, N=top_n, filter_already_liked_items=False)
            top_items, prob = zip(*recommendations)
            pred_item = np.append(pred_item, top_items)   
            pred_prob = np.append(pred_prob, prob)
            pred_user = np.append(pred_user, np.full(shape=len(prob), fill_value=user, dtype=np.int))         

        all_multinomials = coo_matrix((pred_prob, (pred_user, pred_item)), shape=(num_users, num_items)).toarray()

        if save_multinomials:
            save_pickle(filename, all_multinomials, False)

    return all_multinomials



def save_model_multinomials():
    
    data_dir = DATA_DIR
    results_dir = os.path.join(MODEL_DIR)    
    
    model_type = 'bpr_model'
    model_results(datasets, train_bpr_model, model_type, data_dir, results_dir, eval_lst,
                        save_multinomials=True, overwrite=False)

    model_type = 'mixture_model'
    model_results(datasets, train_mixture_model, model_type, data_dir, results_dir, eval_lst,
                        save_multinomials=True, overwrite=False)

    model_type = 'mixture_decay_model'
    model_results(datasets, train_mixture_model, model_type, data_dir, results_dir, eval_lst,
                        n_components={'MFP47K_last basket': 0.8}, save_multinomials=True, overwrite=False)

    model_type = 'fpmc_model'
    param = load_pickle(os.path.join(PARAM_DIR, model_type))
    model_results(datasets, train_fpmc_model, model_type, data_dir, results_dir, eval_lst,
                        n_components=param, save_multinomials=True, overwrite=False)

    model_type = 'nmf_model'
    model_results(datasets, train_nmf_model, model_type, data_dir, results_dir, eval_lst,
                        save_multinomials=True, overwrite=False)

    model_type = 'lda_model'
    model_results(datasets, train_lda_model, model_type, data_dir, results_dir, eval_lst,
                        save_multinomials=True, overwrite=False)

    model_type = 'global_model'
    model_results(datasets, train_global_model, model_type, data_dir, results_dir, eval_lst,
                        save_multinomials=True, overwrite=False)
    
    model_type = 'personal_model'
    model_results(datasets, train_favourite_model, model_type, data_dir, results_dir, eval_lst, 
                    save_multinomials=True, overwrite=False)

    model_type = 'wrmf_model'
    param = load_pickle(os.path.join(PARAM_DIR, model_type))
    model_results(datasets, train_wrmf_model, model_type, data_dir, results_dir, eval_lst,
                        n_components=param, save_multinomials=True, overwrite=False)




def train_mixture_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                        num_proc=None, model_type='mixture_model'):
    """
    Runs the Mixture experiment with the original paper implementation. It evaluates on the test set.
    See https://github.com/UCIDataLab/repeat-consumption.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param num_proc: Number of processes to be used. If none, all the processors in the machine will be used.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    filename = os.path.join(results_dir, model_type, dataset_name, 'user_multinomials.pkl')
    if os.path.exists(filename) and not overwrite:
        user_multinomials = load_pickle(filename)
    else:
        weight_filename = os.path.join(results_dir, 'mixture_model', dataset_name, 'mixing_weights.pkl')
        if os.path.exists(weight_filename) and not overwrite:
            # an array of mixing weights, which is n_users x 2 (2 components, self and global)
            mix_weights = load_pickle(weight_filename)
        else:
            train_matrix, global_matrix = get_train_global(train, val, test)
            components = [train_matrix, global_matrix]  # can add more components here
            mix_weights = learn_mixing_weights(components, val, num_proc)
            if save_multinomials:
                save_pickle(weight_filename, mix_weights, False)

        user_multinomials = _user_multinomials(train, val, test, mix_weights)

        if save_multinomials:
            save_pickle(filename, user_multinomials, False)

    return user_multinomials
