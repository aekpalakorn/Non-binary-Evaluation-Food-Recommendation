import os
import numpy as np
from src.io import load_pickle

PROJECT_DIR = ''
FIG_DIR = os.path.join(PROJECT_DIR, 'figure')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')
DIARY_DIR = os.path.join(DATA_DIR, 'UserStudy2', 'diary')


# User study 1, worker ratings
mturk_rating_file = os.path.join(DATA_DIR, 'UserStudy1', 'mturk_worker_ratings.csv')
# User study 1, SimJudgement by majority voting 
mturk_simjudgement_file = os.path.join(DATA_DIR, 'UserStudy1', 'simjudgement.csv')
# User study 1, SimJudgement by majority voting + similarity scores
mturk_substitution_score_file = os.path.join(DATA_DIR, 'UserStudy1', 'simjudgement_scores.csv')

# Raw substitution ratings
substitution_rating_file = os.path.join(DATA_DIR, 'UserStudy2', 'substitution_ratings.csv')
# Raw substitution ratings + similarity scores
substitution_score_file = os.path.join(DATA_DIR, 'UserStudy2', 'substitution_scores.csv')

# Raw preference ratings
preference_rating_file = os.path.join(DATA_DIR, 'UserStudy2', 'preference_ratings.csv')
# Basket-wise mean preference ratings and scores@10
preference_score_file = os.path.join(DATA_DIR, 'UserStudy2', 'preference_scores.csv')

# Intermediate files, item level score, to be consolidated into preference_score_file 
preference_rating_scores = {'BLEU': os.path.join(DATA_DIR, 'UserStudy2', 'preference_BLEU.csv'),
                            'ROUGE': os.path.join(DATA_DIR, 'UserStudy2', 'preference_ROUGE.csv'),
                            'BERTScores': os.path.join(DATA_DIR, 'UserStudy2', 'preference_BERTScores.csv'),
                            'hP': os.path.join(DATA_DIR, 'UserStudy2', 'preference_hP.csv'),
                            'hR': os.path.join(DATA_DIR, 'UserStudy2', 'preference_hR.csv'),                              
                            'hP-Sim': os.path.join(DATA_DIR, 'UserStudy2', 'preference_hP-Sim.csv'),
                            'hR-Sim': os.path.join(DATA_DIR, 'UserStudy2', 'preference_hR-Sim.csv')}

# temp files
name_ref = os.path.join(TEMP_DIR, 'UserStudy1', "full_name_ref.csv")
user_ground_truth = os.path.join(TEMP_DIR, 'UserStudy1', 'selected_user_ground_truth.csv')
user_recommendations = os.path.join(TEMP_DIR, 'UserStudy1', 'selected_user_recommendations.csv')
mturk_rating_raw = os.path.join(TEMP_DIR, 'UserStudy1', 'mturk_qn1.csv')
mturk_rating_filter = os.path.join(TEMP_DIR, 'UserStudy1', 'selected_pairs_fingerprint.npy')
label_file = os.path.join(TEMP_DIR, 'label_summary.pkl')
label_file_full = os.path.join(TEMP_DIR, 'label_summary_55.pkl')
lst_ref = load_pickle(os.path.join(TEMP_DIR, 'tag_index.pkl'))
weighing_scheme = {'freq': load_pickle(os.path.join(TEMP_DIR, 'weight_100freq.pkl')), 
                   '124':  load_pickle(os.path.join(TEMP_DIR, 'weight_124.pkl')), 
                   'equal': np.ones(len(lst_ref))}
BERT_F1_dict_file = os.path.join(TEMP_DIR, 'BERT_F1.pkl')
FNAMES_BERT_SCORES_FILE = os.path.join(TEMP_DIR, 'fnames_BERT.pkl')
WV_dict_file = os.path.join(TEMP_DIR, 'WV_F1.pkl')

# Same numbers as in substitution_score_file
substitution_rating_scores = {'BLEU': os.path.join(TEMP_DIR, 'UserStudy2', 'substitution_BLEU.csv'),
                              'ROUGE': os.path.join(TEMP_DIR, 'UserStudy2', 'substitution_ROUGE.csv'),
                              'BERTScores': os.path.join(TEMP_DIR, 'UserStudy2', 'substitution_BERTScores.csv'),
                              'hMatch': os.path.join(TEMP_DIR, 'UserStudy2', 'substitution_hMatch.csv'),
                              'hSim': os.path.join(TEMP_DIR, 'UserStudy2', 'substitution_hSim.csv')}
# Same numbers as in substitution_score_file
mturk_substitution_rating_scores = {'BLEU': os.path.join(TEMP_DIR, 'UserStudy1', 'substitution_BLEU.csv'),
                              'ROUGE': os.path.join(TEMP_DIR, 'UserStudy1', 'substitution_ROUGE.csv'),
                              'BERTScores': os.path.join(TEMP_DIR, 'UserStudy1', 'substitution_BERTScores.csv'),
                              'hMatch': os.path.join(TEMP_DIR, 'UserStudy1', 'substitution_hMatch.csv'),
                              'hSim': os.path.join(TEMP_DIR, 'UserStudy1', 'substitution_hSim.csv')}

user_name_file = os.path.join(TEMP_DIR, 'UserStudy2', 'username.npy')
with open(user_name_file, 'rb') as f:
    username = np.load(f)
    
qn_model = {1: 'MixtureTW', 2: 'SASRec', 3: 'NMF', 4: 'Personal', 5: 'FPMC'}
models = ['Personal', 'MixtureTW', 'NMF', 'FPMC', 'SASRec']


    
