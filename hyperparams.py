import numpy as np
import pandas as pd
import itertools as it
import pickle
from sklearn.model_selection import cross_val_score

def init_clf(clf_code, clf_dict):
    if clf_code in clf_dict:
        return(clf_dict[clf_code])
    else:
        raise ValueError('You are probably trying to use a classifier that you haven\'t implemented yet!')
        return(None)

def compute_accuracies(data_X_train, data_y_train, data_X_test, data_y_test, clf_code_list, learned_hyperparams_dict):
    
    accuracy_dict = dict()
    
    for clf_code in clf_code_list:
        clf = init_clf(clf_code)
        learned_hyperparams = learned_hyperparams_dict[clf_code]
        
        for param_name, param_val in learned_hyperparams.items():
            setattr(clf, param_name, param_val)

        clf.fit(data_X_train, data_y_train)

        train_preds = clf.predict(data_X_train)
        test_preds  = clf.predict(data_X_test)

        train_acc   = sum(train_preds == data_y_train)/len(data_y_train)
        test_acc    = sum(test_preds == data_y_test)/len(data_y_test)
        
        accuracy_dict[clf_code] = train_acc, test_acc
    
    return(accuracy_dict)

class hyperparam_learner:
    # TODO: TAKE CARE OF CLF_DICT!!!

    data_X_train  = None
    data_y_train  = None
    data_desc     = None
    clf_code_list = None
    cv_fold       = None
    
    hyperparam_dict = None
    
    is_fitted     = False
    has_learned   = False
    
    @classmethod
    def load(cls, filename):
        with open(filename + '.pkl', 'rb') as f:
            return pickle.load(f)
    
    def save(self, filename):
        if not self.is_fitted:
            raise Exception('I am not fitted yet!')
        if not self.has_learned:
            raise Exception('I have not learned any hyperparameters yet!')
        
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def fit(self, data_X_train, data_y_train, data_desc, clf_code_list, cv_fold):
        self.data_X_train   = data_X_train
        self.data_y_train   = data_y_train
        self.data_desc      = data_desc
        self.clf_code_list  = clf_code_list
        self.cv_fold        = cv_fold
        self.is_fitted      = True
    
    def learn_hyperparams(self, clf_param_grid):
        if(not self.is_fitted):
            raise Exception('I am not fitted yet!')
            
        if isinstance(self.clf_code_list, str):
            if self.clf_code_list == 'all':
                self.clf_code_list = list(CLF_DICT.keys())
            else:
                self.clf_code_list = [self.clf_code_list]

        learned_hyperparam_dict = dict()
        considered_hyperparam_dict = dict()

        for clf_code in self.clf_code_list:
            considered_hyperparam_dict[clf_code] = clf_param_grid[clf_code]
            gs_clf = GridSearchCV(init_clf(clf_code), clf_param_grid[clf_code], cv=self.cv_fold, n_jobs=4)
            gs_clf.fit(self.data_X_train, self.data_y_train)
            learned_hyperparam_dict[clf_code] = gs_clf.best_params_

        hyperparam_dict = {'learned_hyperparams'    : learned_hyperparam_dict,
                           'considered_hyperparams' : considered_hyperparam_dict,
                           'data_desc'              : self.data_desc}
        
        self.hyperparam_dict = hyperparam_dict
        self.has_learned = True
        return(self.hyperparam_dict)
    
    def get_hyperparams(self):
        if(self.has_learned):
            return(self.hyperparam_dict)
        else:
            raise Exception('I have not learned any hyperparameters yet!')

class hyperparam_explorer:
    """
    Calculates cross-validated scores for combinations of hyperparameters.

        data_dict:        Dict of data (X and y), e.g.:
                          data_dict = {'data1_name': [data1_X_train, data2_y_train],
                                       'data2_name': [data2_X_train, data2_y_train]}.

        hyperparam_dict:  Dict of classifiers and hyperparameters, e.g.:
                          hyperparam_dict = {'knn': {'n_neighbors': [1, 2, 3]},
                                             'svm': {'C': [1e-2, 1e-1],
                                                     'alpha': [1e-5, 1e-6]}}
	
	clf_dict:         Dict of classifiers, e.g.:
			  clf_dict = {'knn': neighbors.KNeighborsClassifier()}

	cv:               Number of cross-validation folds

    Format of returned results object: e.g.,

    {'data1_name': {'clf1_name': clf1_result_matrix,
                    'clf2_name': clf2_result_matrix}
     'data2_name': {'clf1_name': clf1_result_matrix,
                    'clf2_name': clf2_result_matrix}}
    Where the first k columns of the result matrices for each classifier include all possible
      combinations of hyperparameters, the the next column contains the mean cross-validated score
      and the last column contains the sd of cross-validated scores.
      
    Methods:
    
        work():           Explores hyperparameters
        
        get_results():    Returns hyperparameters
        
        save():           Writes object to file (including data, parameters, and results)
        
        load(filename):   Loads object from file (class method)
    """
    
    data_dict       = None
    hyperparam_dict = None
    cv              = None
    clf_dict        = None
    
    has_worked      = False
    
    result_dict     = None
    
    def __init__(self, data_dict, hyperparam_dict, clf_dict, cv=5):
        self.data_dict        = data_dict
        self.hyperparam_dict  = hyperparam_dict
        self.cv               = cv
        self.clf_dict         = clf_dict
        
    def work(self):
        # Report progress so we know when to go and get coffee
        n_hyperparam_list = []
        for clf in self.hyperparam_dict:
            for hyperparam in self.hyperparam_dict[clf]:
                n_hyperparam_list.append(len(self.hyperparam_dict[clf][hyperparam]))
        n_hyperparam_combs = np.product(n_hyperparam_list)
        verbose_n_iter = len(self.data_dict) * n_hyperparam_combs
        verbose_iter = 0

        result_dict = dict()
        # Loop through data sets
        for data_name, data_X_y in self.data_dict.items():
            result_dict[data_name] = dict()
            X = data_X_y[0]
            y = data_X_y[1]

            # Loop through classifiers
            for clf_name, clf_hyperparam_dict in self.hyperparam_dict.items():        
                # Combinations of hyperparameters
                clf_hyperparam_list = list(clf_hyperparam_dict.keys())
                combinations = it.product(*(clf_hyperparam_dict[Name] for Name in clf_hyperparam_list))
                clf_hyperparam_comb_list = list(combinations) # order within tuples as specified in self.hyperparam_dict, use clf_hyperparam_list for indexing hyperparams

                # Scores result matrix. Shape: (num_combinations, num_hyperparams+2) where second last column is scores mean and last column is scores mean
                clf_hyperparam_comb_scores = np.empty((len(clf_hyperparam_comb_list), len(clf_hyperparam_list)+2)) # +2 for scores mean, sd

                # Loop through hyperparameter combinations
                for clf_hyperparam_comb_idx, clf_hyperparam_comb in enumerate(clf_hyperparam_comb_list):
                    clf = init_clf(clf_name, self.clf_dict)

                    # Assign hyperparameters to clf: loop through hyperparameters
                    for hyperparam_name in clf_hyperparam_list:
                        # Find index of hyperparam in clf_hyperparam_comb_list element tuples
                        hyperparam_ind = np.where(np.array(clf_hyperparam_list) == hyperparam_name)[0][0]
                        setattr(clf, hyperparam_name, clf_hyperparam_comb[hyperparam_ind])

                    # Progress report part
                    verbose_iter = verbose_iter + 1
                    print("Working on part {} of {}...".format(verbose_iter, verbose_n_iter))

                    scores = cross_val_score(clf, X, y, cv=self.cv, n_jobs=4)
                    scores_mean = np.mean(scores)
                    scores_sd = np.std(scores)

                    for clf_hyperparam_val_idx, clf_hyperparam_val in enumerate(clf_hyperparam_comb):
                        # Write hyperparam values to matrix
                        clf_hyperparam_comb_scores[clf_hyperparam_comb_idx, clf_hyperparam_val_idx] = clf_hyperparam_val

                        # Write scores mean, sd to matrix
                        clf_hyperparam_comb_scores[clf_hyperparam_comb_idx, -2] = scores_mean
                        clf_hyperparam_comb_scores[clf_hyperparam_comb_idx, -1] = scores_sd

                clf_hyperparam_comb_scores = pd.DataFrame(clf_hyperparam_comb_scores)
                clf_hyperparam_comb_scores.columns = clf_hyperparam_list + ['scores_mean', 'scores_sd']
                result_dict[data_name][clf_name] = clf_hyperparam_comb_scores
        
        self.has_worked = True
        self.result_dict = result_dict
        return(result_dict)
    
    def get_results(self):
        if(not self.has_worked):
            raise Exception("This hyperparameter explorer has not done any exploration work yet!")
        return(self.result_dict)
    
    @classmethod
    def load(cls, filename):
        with open(filename + '.pkl', 'rb') as f:
            return pickle.load(f)
    
    def save(self, filename):
        if not self.has_worked:
            raise Exception("This hyperparameter explorer has not done any exploration work yet!")
        
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
