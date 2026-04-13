import numpy as np



def get_mlc(n_features, n_labels, weighted_instances_handler = False, only_multiclass_classifiers = False):
    payoff_function = np.array(['Accuracy', '\'Jaccard index\'', '\'Hamming score\'', '\'Exact match\'',
                    '\'Jaccard distance\'', '\'Hamming loss\'', '\'ZeroOne loss\'',
                    '\'Harmonic score\'', '\'One error\'', '\'Rank loss\'', '\'Avg precision\'',
                    '\'Log Loss (lim. L)\'', '\'Log Loss (lim. D)\'', '\'Micro Precision\'',
                    '\'Micro Recall\'', '\'Macro Precision\'', '\'Macro Recall\'',
                    '\'F1 (micro averaged)\'', '\'F1 (macro averaged by example)\'',
                    '\'F1 (macro averaged by label)\'', '\'AUPRC (macro averaged)\'',
                    '\'AUROC (macro averaged)\'', '\'Levenshtein distance\''])
    
    MLC_config_dic = {


        'meka.classifiers.multilabel.MULAN.MLkNN': {
            '-normalize': np.array([False, True], dtype=bool),
            '-numOfNeighbors': np.arange(1, 65, dtype=int)
        },

        'meka.classifiers.multilabel.MULAN.HOMER': {
            '-method': np.array(['BalancedClustering', 'Clustering', 'Random']),
            '-clusters': np.arange(2, n_labels+1 if n_labels < 9 else 8, dtype=int),
            '-mll': np.array(['BinaryRelevance', 'ClassifierChain', 'LabelPowerset'])
        },


        'meka.classifiers.multilabel.MULAN.ECC': {
        },


        'meka.classifiers.multilabel.BPNN': {
            '-normalize': np.array([False, True], dtype=bool),
            '-E': np.arange(10, 1001, dtype=int),
            '-H': np.arange(int(0.2 * n_features), n_features + 1, dtype=int),
            '-r': np.around(np.arange(0.001, 0.1001, 0.001), 3),
            '-m': np.around(np.arange(0.1, 0.9, 0.1), 1),
        },
        

        'meka.classifiers.multilabel.BR': {
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)  
        },


        'meka.classifiers.multilabel.CC': {
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)  
        },


        'meka.classifiers.multilabel.RAkEL': {
            '-N': np.arange(0, 6, dtype=int),
            '-P': np.arange(1, 6, dtype=int),
            '-k': np.arange(1, int(n_labels/2) + 1, dtype=int),
            '-M': np.arange(2, min(2 * n_labels, 100) + 1, dtype=int),
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True)
        },
        
        
        'meka.classifiers.multilabel.LC': { # LP
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True)    
        },


        'meka.classifiers.multilabel.BCC': {
            '-X': np.array(['C', 'I', 'Ib', 'Ibf', 'H', 'Hbf', 'X', 'F', 'L', 'None']), 
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },


        'meka.classifiers.multilabel.BRq': {
            '-P': np.around(np.arange(0.1, 0.805, 0.05), 2),
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },
        

        'meka.classifiers.multilabel.FW': {
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },

        'meka.classifiers.multilabel.MCC': {
            '-Iy': np.arange(1, 101, 1, dtype=int),
            '-Is': np.append(np.zeros(1499, dtype=int), np.arange(2, 1501, 1, dtype=int)),
            '-P': payoff_function,            
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },
        

        'meka.classifiers.multilabel.PCC': {     
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        },


        'meka.classifiers.multilabel.PS': {    
            '-P': np.arange(1, 6, dtype=int),
            '-N': np.arange(0, 6, dtype=int),
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },
        

        'meka.classifiers.multilabel.PSt': {    
            '-P': np.arange(1, 6, dtype=int),
            '-N': np.arange(0, 6, dtype=int),
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },


        'meka.classifiers.multilabel.RAkELd': {    
            '-P': np.arange(1, 6, dtype=int),
            '-N': np.arange(0, 6, dtype=int),
            '-batch-size': np.arange(2, min(2 * n_labels, 100) + 1, dtype=int), 
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },


        'meka.classifiers.multilabel.RT': {     
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers=True) 
        },


        'meka.classifiers.multilabel.CT': {
            '-X': np.array(['C', 'I', 'Ib', 'Ibf', 'H', 'Hbf', 'X', 'F', 'L', 'None']), 
            '-Iy': np.arange(1, 101, 1, dtype=int),
            '-Is': np.append(np.zeros(1499, dtype=int), np.arange(2, 1501, 1, dtype=int)),
            '-P': payoff_function,
            '-H': np.array([-1, 0, 1]),
            'if': lambda params: {'-L': np.arange(1, np.sqrt(n_labels) + 2, dtype=int)} if params['-H']==-1 else None,
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)
        },


        'meka.classifiers.multilabel.CDN': {
            '-Ic': np.arange(1, 101, 1, dtype=int),
            '-I': np.arange(100, 1001, 1, dtype=int),
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)
        },
        

        'meka.classifiers.multilabel.CDT': {
           '-H': np.array([-1, 0, 1]),
            '-X': np.array(['C', 'I', 'Ib', 'Ibf', 'H', 'Hbf', 'X', 'F', 'L', 'None']), 
            '-I': np.arange(1, 1001, 1, dtype=int),
            '-Ic': np.arange(1, 101, 1, dtype=int),
            'if': lambda params: {'-L': np.arange(1, np.sqrt(n_labels) + 2, dtype=int)} if params['-H']==-1 else None,
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers)
        },
        
         
        'meka.classifiers.multilabel.PMCC': {
            '-Iy': np.arange(1, 101, 1, dtype=int),
            '-Is': np.arange(50, 1501, 1, dtype=int), # chain iterations
            '-B': np.arange(0.01, 1, 0.01),
            '-O': np.array([0, 1], dtype=int),
            '-M': np.arange(1, 51, dtype=int), # population size
            '-P': payoff_function,            
            '-W': get_slc(n_features, n_labels, weighted_instances_handler, only_multiclass_classifiers) 
        }

    }
    
    return MLC_config_dic


def get_slc(n_features, n_labels, weighted_instances_handler=False, only_multiclass_classifiers=False, randomizable=False):
    
    SLC_config_dic = {

        'weka.classifiers.bayes.NaiveBayes': {
            '-normalize': np.array([False], dtype=bool),
            '-K': np.array([False, True], dtype=bool),
            'if': lambda params: {'-D': np.array([False, True], dtype=bool)} if params['-K']==False else None
        }, 


        'weka.classifiers.rules.PART': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-B': np.array([False, True], dtype=bool),
            '-R': np.array([False, True], dtype=bool),
            'if': lambda params: {'-N': np.array([2,3,4,5])} if params['-R']==True else None
        }, 
        
        
        'weka.classifiers.trees.J48': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-B': np.array([False, True], dtype=bool),
            '-J': np.array([False, True], dtype=bool),
            '-A': np.array([False, True], dtype=bool),
            '-U': np.array([False, True], dtype=bool), 
            'if': lambda params: {'-C': np.arange(0.05, 1, 0.05), '-O': np.array([False, True], dtype=bool), '-S': np.array([False, True], dtype=bool)} if params['-U']==False else None 
        },
         
         
        'weka.classifiers.lazy.IBk': {
            '-normalize': np.array([False, True], dtype=bool),
            '-K': np.arange(1, 65, dtype=int),
            '-X': np.array([False, True]),
            '-I': np.array([False, True]),
            'if': lambda params: {'-F': np.array([False, True])} if params['-I']==False else None
        },
         
         
        'weka.classifiers.functions.SMO': {
            '-normalize': np.array([False, True], dtype=bool),
            '-C': np.arange(0.5, 1.6, 0.1),
            '-N': np.array([0,1,2]),
            '-M': np.array([False, True], dtype=bool),
            '-K': get_kernels_smo()
        },
         
         
        'weka.classifiers.rules.JRip': {
            '-normalize': np.array([False], dtype=bool),
            '-N': np.around(np.arange(1, 5.1, 0.1), 1),
            '-E': np.array([False, True], dtype=bool),
            '-P': np.array([False, True], dtype=bool),
            '-O': np.arange(1, 6, dtype=int)
        },                
         

        'weka.classifiers.functions.Logistic': {
            '-normalize': np.array([False], dtype=bool),
            '-R': np.array([10**(-x) for x in range(12, -2,-1)])
        },
         
         
        'weka.classifiers.bayes.BayesNet': {
            '-normalize': np.array([False], dtype=bool),
            '-D': np.array([True]), 
            '-Q': np.array(['weka.classifiers.bayes.net.search.local.TAN', 
                            'weka.classifiers.bayes.net.search.local.K2 -- -P 1', # INT_P = 1 (Maximum number of parents)
                            'weka.classifiers.bayes.net.search.local.HillClimber -- -P 1', # INT_P = 1
                            'weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1', # INT_P = 1
                            'weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1']) # INT_P = 1
        },
        

        'weka.classifiers.trees.RandomForest': {
            '-normalize': np.array([False], dtype=bool),
            '-I': np.arange(2, 257, dtype=int),
            '-K': np.append(np.zeros(31, dtype=int), np.arange(2, 33, dtype=int)),
            '-depth': np.append(np.zeros(19, dtype=int), np.arange(2, 21, dtype=int))
        },

         
        'weka.classifiers.rules.DecisionTable': {
            '-normalize': np.array([False], dtype=bool),
            '-E': np.array(['acc', 'rmse', 'mae', 'auc']),
            '-I': np.array([False, True], dtype=bool),
            '-S': np.array(['BestFirst', 'GreedyStepwise']),
            '-X': np.array([1,2,3,4])
        },
         
         
        'weka.classifiers.lazy.KStar': {
            '-normalize': np.array([False, True], dtype=bool),
            '-B': np.arange(1, 101, dtype=int),
            '-E': np.array([False, True], dtype=bool),
            '-M': np.array(['a', 'd', 'm', 'n'])
        },
         
         
        'weka.classifiers.trees.LMT': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-B': np.array([False, True], dtype=bool),
            '-R': np.array([False, True], dtype=bool),
            '-C': np.array([False, True], dtype=bool),
            '-P': np.array([False, True], dtype=bool),
            '-W': np.arange(0, 1.01, 0.01),
            '-A': np.array([False, True], dtype=bool),
        },
         
         
        'weka.classifiers.functions.MultilayerPerceptron': {
            '-normalize': np.array([False, True], dtype=bool),
            '-L': np.around(np.arange(0.1, 1.1, 0.1), 1),
            '-M': np.around(np.arange(0, 1.1, 0.1), 1),
            '-H': np.array([int(np.around((n_features + n_labels)/2)), n_features, n_labels, n_features + n_labels]),
            '-B': np.array([False, True], dtype=bool),
            '-R': np.array([False, True], dtype=bool),
            '-D': np.array([False, True], dtype=bool)
        },
         
         
        'weka.classifiers.trees.REPTree': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-L': np.append(-np.ones(19, dtype=int), np.arange(2, 21, dtype=int)),
            '-P': np.array([False, True], dtype=bool)
        },
         
         
        'weka.classifiers.functions.SGD': {
            '-normalize': np.array([False], dtype=bool),
            '-F': np.array([0, 1]), 
            '-L': np.array([10**(x) for x in range(-5, 0, 1)]),
            '-R': np.array([10**(x) for x in range(-12, 2, 1)]),
            '-N': np.array([False, True], dtype=bool),
            '-M': np.array([False, True], dtype=bool)
        },
         
         
        'weka.classifiers.trees.RandomTree': {
            '-normalize': np.array([False], dtype=bool),
            '-M': np.arange(1, 65, dtype=int),
            '-K': np.append(np.zeros(31, dtype=int), np.arange(2, 33, dtype=int)),
            '-depth': np.append(np.zeros(19, dtype=int), np.arange(2, 21, dtype=int)),
            '-N': np.append(np.zeros(4, dtype=int), np.arange(2, 6, dtype=int)),
        },
         
         
        'weka.classifiers.functions.SimpleLogistic': {
            '-normalize': np.array([False], dtype=bool),
            '-W': np.arange(0, 1.1, 0.1),
            '-S': np.array([False, True], dtype=bool),
            '-A': np.array([False, True], dtype=bool),
        },
         
         
        'weka.classifiers.functions.VotedPerceptron': {
            '-normalize': np.array([False, True], dtype=bool),
            '-I': np.arange(1, 10, dtype=int),
            '-M': np.arange(5000, 50001, dtype=int),
            '-E': np.arange(0.2, 5, 0.1)
        },
         
    }
    
    
    if randomizable == True:
        SLC_config_dic = {k: v for k, v in SLC_config_dic.items() if k in [
            'weka.classifiers.trees.RandomForest',
            'weka.classifiers.trees.RandomTree',
            'weka.classifiers.trees.REPTree',
            'weka.classifiers.functions.SGD',
            'weka.classifiers.functions.MultilayerPerceptron']} 
        
        
    if weighted_instances_handler == True:
        SLC_config_dic = {k: v for k, v in SLC_config_dic.items() if k not in [
            'weka.classifiers.trees.LMT',
            'weka.classifiers.lazy.KStar',
            'weka.classifiers.functions.SGD', 
            'weka.classifiers.rules.OneR',
            'weka.classifiers.functions.VotedPerceptron']}
        
        
    if only_multiclass_classifiers == True:  
        SLC_config_dic = {k: v for k, v in SLC_config_dic.items() if k not in [
            'weka.classifiers.functions.SGD', 
            'weka.classifiers.functions.VotedPerceptron']}
        
        
    return SLC_config_dic


def get_kernels_smo():
    config_smo = {
        
        'weka.classifiers.functions.supportVector.PolyKernel': {
            '-E': np.arange(0.2, 5.1, 0.1),
            '-L': np.array([False, True], dtype=bool)
        },
        
        'weka.classifiers.functions.supportVector.NormalizedPolyKernel': {
            '-E': np.arange(0.2, 5.1, 0.1),
            '-L': np.array([False, True], dtype=bool)
        },
        
        'weka.classifiers.functions.supportVector.Puk': {
            '-O': np.arange(0.1, 1.1, 0.1),
            '-S': np.arange(0.1, 10.1, 0.1)
        },
        
        'weka.classifiers.functions.supportVector.RBFKernel': {
            '-G': np.array([10**(x) for x in range(-4, 1, 1)])
        }
        
    }
        
    return config_smo
