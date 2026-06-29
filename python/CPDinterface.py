#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:54:04 2023

@author: rve
"""


import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics.cluster import rand_score
from sklearn.base import BaseEstimator
from scipy.signal import find_peaks
import itertools
# rpy2 is only required for R‑based algorithms. allow import failure when R not installed.
try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
except ImportError:
    importr = None
    numpy2ri = None
# from rpy2.robjects.methods import RS4



# r_obj must be RS4 object
def getContentRS4(r_obj):
    # Access the slot names
    slot_names = r_obj.slotnames()

    # Retrieve slot values using do_slot() method
    slot_values = {name: numpy2ri.rpy2py(r_obj.do_slot(name)) for name in slot_names}
    
    return slot_values


# ** I am considering that trueCPs and detectedCPs are sorted in the increase order **
# Papers where I got the metrics:
# A survey of methods for time series change point detection, 2017
# Real-time change point detection with application to smart home time series data, 2019
# Selective review of offline change point detection methods, 2020
# An Evaluation of Change Point Detection Algorithms, 2022
# Note: we can use any metric that compares two clustering solutions (such as the rand_score)
class CPDmetrics:
    def __init__(self, trueCPs, detectedCPs, signal, delta = 1):
        self.trueCPs = trueCPs
        self.detectedCPs = detectedCPs
        self.signal = signal
        self.numberOfInstances = signal.shape[0]
        self.delta = delta
        
        if isinstance(signal, pd.Series):
            self.indexes = signal.index
        else:
            self.indexes = np.array(range(self.numberOfInstances))
        
        self.tn, self.fp, self.fn, self.tp = self.__compareCP()
        self.trueSegments = self.__get_segments_indexes_aux(self.trueCPs)
        self.detectedSegments = self.__get_segments_indexes_aux(self.detectedCPs)
        self.trueClusters = self.__get_clusters_aux(self.trueSegments)
        self.detectedClusters = self.__get_clusters_aux(self.detectedSegments)        
        
    def __compareCP(self):
        trueCPs = self.trueCPs
        detectedCPs = self.detectedCPs
        numberOfInstances = self.numberOfInstances
        delta = self.delta
        
        # helper to compute absolute distance between two change point values
        def diff(a, b):
            # handle timestamps vs numeric indices
            if hasattr(a, 'timestamp') or isinstance(a, pd.Timestamp) or isinstance(a, pd.DatetimeIndex):
                a_ts = pd.Timestamp(a)
                b_ts = pd.Timestamp(b)
                return abs(a_ts - b_ts)
            else:
                return abs(a - b)

        tp = fp = fn = 0
        i1 = i2 = 0
        while i1 < len(detectedCPs) and i2 < len(trueCPs):
            data1 = detectedCPs[i1]
            data2 = trueCPs[i2]
            if data1 < data2:
                distance = diff(data2, data1)
                # determine comparison value depending on delta type
                if isinstance(delta, (pd.Timedelta, timedelta)):
                    # if distance is numeric convert delta to comparable days
                    if isinstance(distance, (int, float, np.integer)):
                        if isinstance(delta, pd.Timedelta):
                            delta_cmp = int(delta / pd.Timedelta(days=1))
                        else:
                            delta_cmp = delta.days
                    else:
                        delta_cmp = delta
                else:
                    delta_cmp = delta
                good = distance <= delta_cmp
                if good:
                    tp = tp +1
                    i2 = i2 + 1
                else:
                    fp = fp + 1
                i1 = i1 + 1
            elif data2 < data1:
                distance = diff(data1, data2)
                if isinstance(delta, (pd.Timedelta, timedelta)):
                    if isinstance(distance, (int, float, np.integer)):
                        if isinstance(delta, pd.Timedelta):
                            delta_cmp = int(delta / pd.Timedelta(days=1))
                        else:
                            delta_cmp = delta.days
                    else:
                        delta_cmp = delta
                else:
                    delta_cmp = delta
                good = distance <= delta_cmp
                if good:
                    tp = tp + 1
                    i1 = i1 + 1
                i2 = i2 + 1
            else:
                tp = tp + 1
                i1 = i1 + 1
                i2 = i2 + 1
        fp = fp + (len(detectedCPs) - i1)
        fn = len(trueCPs) - tp
        tn = numberOfInstances - tp - fn - fp
        return tn, fp, fn, tp

    def get_confusion_matrix_ravel(self):
        return self.tn, self.fp, self.fn, self.tp
    
    def get_true_positive_rate(self):
        if self.tp == 0: return 0
        return self.tp / (self.tp + self.fn)
    
    def get_false_positive_rate(self):
        if self.fp == 0: return 0
        return self.fp / (self.fp + self.tn)
    
    def get_precision(self):
        if self.tp == 0: return 0
        return self.tp / (self.tp + self.fp)
    
    def get_recall(self):
        if self.tp == 0: return 0
        return self.tp / (self.tp + self.fn)
    
    def get_f1measure(self):
        precision = self.get_precision()
        recall = self.get_recall()
        if precision == 0 or recall == 0: return 0
        return 2 * precision * recall / (precision + recall)
    
    def get_gmean(self):
        # geometric mean of true positive rate and true negative rate
        try:
            tpr = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
            tnr = self.tn / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
            return (tpr * tnr) ** 0.5
        except Exception:
            return 0
    
    def __get_segments_indexes_aux(self, CPs):
        Segments = []
        segment_begin = self.indexes[0]
        for cp in CPs:
            segment  = np.where((self.indexes >= segment_begin) & (self.indexes < cp))[0]
            Segments.append(segment)
            segment_begin = cp
        Segments.append(np.where((self.indexes >= segment_begin))[0])
        return Segments
    
    def get_segments_indexes(self):
        return self.trueSegments, self.detectedSegments
    
    def __get_jaccard(self, segment1, segment2):
        set1 = set(segment1)
        set2 = set(segment2)
        return len(set1.intersection(set2)) / len(set1.union(set2))
    
    def get_covering(self):
        trueSegments, detectedSegments = self.trueSegments, self.detectedSegments
        soma = 0
        idx_detectedSegments_start = 0
        for trueSegment in trueSegments:
            best_jaccard = 0
            best_jaccard_idx = idx_detectedSegments_start
            for i in range(idx_detectedSegments_start, len(detectedSegments)):
                jaccard = self.__get_jaccard(trueSegment, detectedSegments[i])
                if best_jaccard < jaccard:
                    best_jaccard = jaccard
                    best_jaccard_idx = i
            soma += len(trueSegment) * best_jaccard
            idx_detectedSegments_start = best_jaccard_idx + 1
        return soma / self.numberOfInstances
    
    def __get_clusters_aux(self, Segments):
        Clusters = np.zeros(shape=(self.numberOfInstances,))
        value = 1
        for segment in Segments[1:]:
            Clusters[segment] = value
            value += 1
        return Clusters
    
    def get_clusters(self):
        return self.trueClusters, self.detectedClusters
    
    def get_rand_index(self):
        return rand_score(self.trueClusters, self.detectedClusters)
    
    def get_all_scores(self):
        scores = {'score_tp':self.tp,
                  'score_tn':self.tn,
                  'score_fp':self.fp,
                  'score_fn':self.fn,
                  'score_tpRate':self.get_true_positive_rate(),
                  'score_fpRate':self.get_false_positive_rate(),
                  'score_precision':self.get_precision(),
                  'score_recall':self.get_recall(),
                  'score_f1measure':self.get_f1measure(),
                  'score_gmean':self.get_gmean(),
                  'score_covering':self.get_covering(),
                  'score_randIndex':self.get_rand_index()}
        return scores


class CPDEstimator(BaseEstimator):
    def __init__(self, algo, **kwargs):
        # Initialize your custom parameters
        self.algo = algo
        self.params = kwargs
        self.estimator = None
        self.numberOfSamples_ = None
        self.detectedCPs_ = None
        self.scores_ = None
        
        # print(self.algo)
        # for key, value in kwargs.items():
        #     print("{0} = {1}".format(key, value))

        # Initialize any additional variables or objects needed
        ## None
    
    def set_params(self, **kwargs):
        self.params = kwargs

    def __fit(self, X):
        # Implement the fit method to train your estimator
        # X: Training data
        self.numberOfSamples_ = X.shape[0]

        # Perform the necessary computations or training steps
        if self.algo=='Pelt':
            from ruptures import Pelt
            self.estimator = Pelt(model=self.params['model'], min_size=self.params['min_size'], jump=self.params['jump']).fit(X)
        elif self.algo=='Binseg':
            from ruptures import Binseg
            self.estimator = Binseg(model=self.params['model'], min_size=self.params['min_size'], jump=self.params['jump']).fit(X)
        elif self.algo=='BottomUp':
            from ruptures import BottomUp
            self.estimator = BottomUp(model=self.params['model'], min_size=self.params['min_size'], jump=self.params['jump']).fit(X)
        elif self.algo=='Window':
            from ruptures import Window
            self.estimator = Window(model=self.params['model'], min_size=self.params['min_size'], jump=self.params['jump'], width=self.params['width']).fit(X)
        elif self.algo=='KernelCPD':
            from ruptures import KernelCPD
            self.estimator = KernelCPD(kernel=self.params['kernel'], min_size=self.params['min_size'], jump=self.params['jump']).fit(X)
        elif self.algo=='CUSUM':
            from ocpdet import CUSUM
            self.estimator = CUSUM(k=self.params['k'], h=self.params['h'], burnin=self.params['burnin'], mu=self.params['mu'], sigma=self.params['sigma'])
            self.estimator.process(X)
        elif self.algo=='EWMA':
            from ocpdet import EWMA
            self.estimator = EWMA(r=self.params['r'], L=self.params['L'], burnin=self.params['burnin'], mu=self.params['mu'], sigma=self.params['sigma'])
            self.estimator.process(X)
        elif self.algo=='TwoSample':
            from ocpdet import TwoSample
            self.estimator = TwoSample(statistic=self.params['statistic'], threshold=self.params['threshold'])
            self.estimator.process(X)
        elif self.algo=='bcp': # Package ‘bcp’ was removed from the CRAN repository.
            if importr is None:
                raise ImportError("rpy2 not available for algorithm 'bcp'")
            bcp=importr('bcp')
            self.estimator = bcp.bcp(numpy2ri.numpy2rpy(X), w0=self.params['w0'], p0=self.params['p0'], d=self.params['d'], burnin=self.params['burnin'])
        elif self.algo=='sbs':
            if importr is None:
                raise ImportError("rpy2 not available for algorithm 'sbs'")
            wbs=importr('wbs')
            resp_wbs = wbs.sbs(numpy2ri.numpy2rpy(X))
            th_const_float = float(self.params['th_const'])
            self.estimator = wbs.changepoints_sbs(resp_wbs, th_const=th_const_float, penalty=self.params['penalty'])
        elif self.algo=='wbs':
            if importr is None:
                raise ImportError("rpy2 not available for algorithm 'wbs'")
            wbs=importr('wbs')
            resp_wbs = wbs.wbs(numpy2ri.numpy2rpy(X))
            th_const_float = float(self.params['th_const'])
            self.estimator = wbs.changepoints_wbs(resp_wbs, th_const=th_const_float, penalty=self.params['penalty'])
        elif self.algo=='cpm1B':
            if importr is None:
                raise ImportError("rpy2 not available for algorithm 'cpm1B'")
            cpm=importr('cpm')
            alpha_float = float(self.params['alpha'])
            self.estimator = cpm.detectChangePointBatch(numpy2ri.numpy2rpy(X),self.params['test_statistic'], alpha=alpha_float)
        elif self.algo=='cpm1S':
            if importr is None:
                raise ImportError("rpy2 not available for algorithm 'cpm1S'")
            cpm=importr('cpm')
            ARL0_int = int(self.params['ARL0'])
            startup_int = int(self.params['startup'])
            self.estimator = cpm.detectChangePoint(numpy2ri.numpy2rpy(X),self.params['test_statistic'],ARL0=ARL0_int,startup=startup_int)
        elif self.algo=='cpmMS':
            if importr is None:
                raise ImportError("rpy2 not available for algorithm 'cpmMS'")
            cpm=importr('cpm')
            ARL0_int = int(self.params['ARL0'])
            startup_int = int(self.params['startup'])
            self.estimator = cpm.processStream(numpy2ri.numpy2rpy(X),self.params['test_statistic'],ARL0=ARL0_int,startup=startup_int)
        elif 'SegNeigh' in self.algo:
            if importr is None:
                raise ImportError(f"rpy2 not available for algorithm '{self.algo}'")
            changepoint=importr('changepoint')
            Q_int = int(self.params['Q'])
            pen_value_float = float(self.params['pen_value'])
            if 'MeanVar' in self.algo:
                self.estimator = changepoint.cpt_meanvar(numpy2ri.numpy2rpy(X),penalty=self.params['penalty'],pen_value=pen_value_float,method='SegNeigh',Q=Q_int,test_stat=self.params['test_stat'], minseglen=1)
            elif 'Var' in self.algo:
                self.estimator = changepoint.cpt_var(numpy2ri.numpy2rpy(X),penalty=self.params['penalty'],pen_value=pen_value_float,method='SegNeigh',Q=Q_int,test_stat=self.params['test_stat'], minseglen=1)
            else: # 'Mean'
                self.estimator = changepoint.cpt_mean(numpy2ri.numpy2rpy(X),penalty=self.params['penalty'],pen_value=pen_value_float,method='SegNeigh',Q=Q_int,test_stat=self.params['test_stat'], minseglen=1)
                
        # Return the fitted estimator
        return self

    def __predict(self):
        # Implement the predict method to make predictions
        predictions = None

        # Perform the necessary computations or prediction steps
        if self.algo in ['Pelt', 'KernelCPD']:
            predictions = self.estimator.predict(pen=self.params['pen'])
            predictions = predictions[:-1]
        elif self.algo in ['Binseg', 'BottomUp', 'Window']:
            pen = self.params['pen'] if 'pen' in self.params.keys() else None
            epsilon = self.params['epsilon'] if 'epsilon' in self.params.keys() else None
            predictions = self.estimator.predict(pen=pen, epsilon=epsilon)
            predictions = predictions[:-1]
        elif self.algo in ['CUSUM', 'EWMA', 'TwoSample']:
            predictions = self.estimator.changepoints if len(self.estimator.changepoints) > 0 else []
        elif self.algo=='bcp':
            posteriorProb = numpy2ri.rpy2py(self.estimator[7])
            predictions, _ = find_peaks(posteriorProb, height=self.params['height'], distance=self.params['distance'])
        elif self.algo in ['sbs', 'wbs']:
            cpsth = self.estimator[3]
            aux = numpy2ri.rpy2py(cpsth[0])
            if len(aux) == 1 and aux[0] < 0: aux = []
            predictions = aux.astype('int64') if len(aux) > 0 else []
        elif self.algo=='cpm1B':            
            aux = numpy2ri.rpy2py(self.estimator[2])
            predictions = aux.astype('int64') if len(aux) > 0 else []
        elif self.algo=='cpm1S':
            aux = numpy2ri.rpy2py(self.estimator[3])
            predictions = aux.astype('int64') if len(aux) > 0 else []
        elif self.algo=='cpmMS':
            aux = numpy2ri.rpy2py(self.estimator[1])
            predictions = aux.astype('int64') if len(aux) > 0 else []
        elif 'SegNeigh' in self.algo:
            res_changepoint = getContentRS4(self.estimator)
            aux = res_changepoint['cpts']
            aux = aux[:-1]
            predictions = aux.astype('int64') if len(aux) > 0 else []
        
        # Return the predictions
        return predictions
    
    def fit_predict(self, X):
        # Implement the fit_predict method as a combination of fit and predict
        # X: Training data - numpy.ndarray or pandas Series with shape=(n,)
        
        self.data_ = X
            
        # Call fit method
        if isinstance(X, pd.Series):
            self.__fit(X.to_numpy())
        else:
            self.__fit(X)
        
        # Call predict method
        self.detectedCPs_ = self.__predict()
        
        # Organizing the output
        if isinstance(X, pd.Series):
            self.detectedCPs_ = X.index[self.detectedCPs_]
        if isinstance(self.detectedCPs_, pd.DatetimeIndex):
            self.detectedCPs_ = self.detectedCPs_.sort_values()
        else:
            self.detectedCPs_.sort()
        
        # Return the predictions
        return self.detectedCPs_
    
    def scores(self, trueCPs, delta):
        # Implement the score method to evaluate the model's performance
        metrics = CPDmetrics(trueCPs, self.detectedCPs_, self.data_, delta)
        scores = metrics.get_all_scores()
        self.scores_ = scores
        return scores



class CPDGridSearch:
    def __init__(self, estimator, param_grid, scoring='score_gmean'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.results_names_ = None
        self.results_values_ = None

    def fit(self, X, trueCPs, delta=0):
        best_score = None
        best_params = None
        
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        self.results_values_ = []
        for combination in combinations:
            #print(combination)
            params = dict(zip(param_names, combination))
            self.estimator.set_params(**params)
            self.estimator.fit_predict(X)
            scores = self.__score(trueCPs, delta)
            
            self.results_values_.append(list(params.values()) + list(scores.values()))
            
            score = scores[self.scoring]
            if best_score is None or score > best_score:
                best_score = score
                best_params = params

        self.results_names_ = param_names + list(scores.keys())
        
        self.best_estimator_ = self.estimator
        self.best_estimator_.set_params(**best_params)
        self.best_estimator_.fit_predict(X)
        self.best_params_ = best_params
        self.best_score_ = best_score
        
    def __score(self, trueCPs, delta):
        scores = self.estimator.scores(trueCPs, delta)
        return scores




if __name__ == "__main__":
    ###############################################################################
    # Example usage:
    
    # Dados:    
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=2., scale=1., size=(100,)),
        np.random.normal(loc=5., scale=1., size=(100,)),
        np.random.normal(loc=3., scale=1., size=(100,)),
    ])
    trueCPs = [100, 200]
    delta = 5 # for trueCPs as datetimes, delta must be like that: timedelta(days=numberofDays) - from datetime import timedelta -
    
    
    print('TEST of CPDEstimator class')
    # estimator = CPDEstimator(algo='Pelt', model='l1', min_size=2, jump=1, pen=10)
    # estimator = CPDEstimator(algo='Binseg', model='l1', min_size=2, jump=1, pen=5, epsilon=None)
    # estimator = CPDEstimator(algo='BottomUp', model='l1', min_size=2, jump=1, pen=5, epsilon=None)
    # estimator = CPDEstimator(algo='Window', model='l1', min_size=2, jump=1, width=20, pen=5, epsilon=None)
    # estimator = CPDEstimator(algo='KernelCPD', kernel='rbf', min_size=2, jump=1, pen=10)
    # estimator = CPDEstimator(algo='CUSUM', k=1, h=2., burnin=7, mu=0., sigma=1.)
    # estimator = CPDEstimator(algo='EWMA', r=0.5, L=3., burnin=0, mu=0., sigma=1.)
    # estimator = CPDEstimator(algo='TwoSample', statistic="Kolmogorov-Smirnov", threshold=3.1)
    # estimator = CPDEstimator(algo='bcp', w0=0.2, p0=0.2, d=10, burnin=10, height=0.3, distance=5)
    # estimator = CPDEstimator(algo='sbs', th_const=1.3, penalty="bic.penalty")
    # estimator = CPDEstimator(algo='wbs', th_const=1.3, penalty="bic.penalty")
    # estimator = CPDEstimator(algo='cpm1B', test_statistic="Cramer-von-Mises", alpha=0.05)
    # estimator = CPDEstimator(algo='cpm1S', test_statistic="Student", ARL0=370, startup=20)
    # estimator = CPDEstimator(algo='cpmMS', test_statistic="Kolmogorov-Smirnov", ARL0=370, startup=3)
    estimator = CPDEstimator(algo='SegNeighMean', penalty="None", Q=3, test_stat="Normal", pen_value=0)
    cps = estimator.fit_predict(data)
    print(cps)
    scores = estimator.scores(trueCPs, delta)
    print(scores)
    
    
    # print('TEST of CPDGridSearch class')
    # param_grid = {'model': ['l1', 'l2', 'rbf'], 'min_size': [2], 'jump':[1], 'pen':[5,10]}
    # MyEstimator = CPDEstimator(algo='Pelt')
    # grid_search = CPDGridSearch(estimator=MyEstimator, param_grid=param_grid)
    # grid_search.fit(data, trueCPs, delta)
    # grid_search_results = pd.DataFrame(data=grid_search.results_values_, columns=grid_search.results_names_)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    
    
    
    ###############################################################################
    ##################### GRID PARAMETERS FOR EACH ALGORITHM ######################
    windowSize = 7 # must be defined accordingly to the experimental protocol
    
    ### RUPTURES
    ## model: segment model - cost function
    model = ["l1", "l2", "rbf"]
    ## min_size: minimum segment length => must be defined accordingly to the experimental protocol
    min_size = [windowSize]
    ## jump: subsample (one every *jump* points) => must be = 1 if all data points are candidates to CPs
    jump = [1]
    ##  width: window length => must be defined accordingly to the experimental protocol
    width = min_size
    ## pen: penalty value => must be adapted accordingly to the dataset
    pen=np.logspace(np.log10(np.median(data)/10), np.log10(np.median(data)*10), 10)
    
    param_grid_rup = {'model':model, 'min_size':min_size, 'jump':jump, 'pen':pen} # Pelt, Binseg, BottomUp
    param_grid_Window = {'model':model, 'min_size':min_size, 'jump':jump, 'pen':pen, 'width':width}
    param_grid_KernelCPD = {'kernel':['linear', 'rbf',  'cosine'], 'min_size':min_size, 'jump':jump, 'pen':pen}
    ###
    
    
    ### OCPDET
    ## k : float, default=0.25
       # Control parameter of the CUSUM algorithm monitoring the gap between  the normalised stream and the algorithm statistics. 
    k = np.linspace(0,1,21)
    ## h : float, default=8.
       # Control parameter of the CUSUM algorithm used as a threshold for the  decision rule.
    h = np.linspace(0,10,11)
    ## burnin : int, default=50
       # Number of firts observed values processed before a changepoint can be detected. 
    burnin = [windowSize-1]
    ## mu : float, default=0.
       # Initial mean value of the stream. Recall that CUSUM assumes that observations are normally distributed.
    mu = [0.]
    ## sigma : float, default=1.
       # Initial standard deviation of the stream.
    sigma = [1.]
    ## r : float, default=0.1
       #Control parameter of the EWMA algorithm monitoring the learning rate of the exponentially moving weighted average Z (between 0 and 1).
    r = np.linspace(0,1,21)
    ## L : float, default=2.4.
       # Control parameter of the EWMA algorithm used as a threshold for the  decision rule and controlling the bandwith.
    L = np.linspace(2.4, 3.0, 7)
    ##  statistic : str, default="Lepage"
        # Test statistic to be used by the algorithm. Use 'Mann-Whitney' for changes
        # in the location, 'Mood' for changes in the scale, 'Lepage' for changes in 
        # both location and scale, 'Kolmogorov-Smirnov' and 'Cramer-von-Mises' for 
        # general changes in distribution.
    statistic = ["Lepage", "Mann-Whitney", "Mood", "Kolmogorov-Smirnov", "Cramer-von-Mises"]
    ## threshold : float, default=3.1
       # Threshold value for the test statistic, must be suited for each statistic.
    threshold = np.linspace(2, 4, 21)
    
    param_grid_cusum = {'k':k, 'h':h, 'burnin':burnin, 'mu':mu, 'sigma':sigma}
    param_grid_ewma = {'r':r, 'L':L, 'burnin':burnin, 'mu':mu, 'sigma':sigma}
    param_grid_2sample = {'statistic':statistic, 'threshold':threshold, 'burnin':burnin, 'mu':mu, 'sigma':sigma}
    ###
    
    
    ### BCP
    ## w0 (optional) a single numeric value in the multivariate case or a vector of values in
       # the regression case; in both, the value(s), between 0 and 1, is/are the parameter(s)
       # in the uniform prior(s) on the signal-to-noise ratio(s). If no value is specified,
       # the default value of 0.2 is used, as recommended by Barry and Hartigan (1993).
    w0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] #must be python float, not numpy float
    ## p0 (optional) a value between 0 and 1. For sequential data, it is the parameter of
       # the prior on change point probabilities, U(0, p0), on the probability of a change
       # point at each location in the sequence; for data on a graph, it is the parameter in
       # the partition prior, p0l(ρ) , where l(ρ) is the boundary length of the partition.
       # default value: 0.2
    p0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    ## d (optional) a positive number only used for linear regression change point models.
       # Lower d means higher chance of fitting the full linear model (instead of the
       # intercept-only model); see prior for τS in Wang and Emerson (2015).
       # default value: 10
    d = list(range(3,14))
    
    height = [0.3, 0.4, 0.5] # parameter from scipy.signal.find_peaks
    distance = [5, 6, 7] # parameter from scipy.signal.find_peaks => must be defined accordingly to the experimental protocol
    
    param_grid_bcp = {'w0':w0, 'p0':p0, 'd':d, 'burnin':burnin, 'height':height, 'distance':distance}
    ###
    
    
    ### WBS
    ## th_const: used to calculate the threshold
       # default value: 1.3
    th_const = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    ## penalty: penalty functions to be used
    penalty = ["bic.penalty", "mbic.penalty", "ssic.penalty"]
    
    param_grid_wbs = {'th_const':th_const, 'penalty':penalty}
    ###
    
    
    ### CPM
    ## cpmType: The type of CPM which is used. With the exception of the FET, these CPMs are all implemented in their two sided forms,
              # and are able to detect both increases and decreases in the parameters monitored. Possible arguments are:
              # • Student: Student-t test statistic, as in [Hawkins et al, 2003]. Use to detect mean changes in a Gaussian sequence.
              # • Bartlett: Bartlett test statistic, as in [Hawkins and Zamba, 2005]. Use to detect variance changes in a Gaussian sequence.
              # • GLR: Generalized Likelihood Ratio test statistic, as in [Hawkins and Zamba, 2005b]. Use to detect both mean and variance changes in a Gaussian sequence.
              # • Exponential: Generalized Likelihood Ratio test statistic for the Exponential distribution, as in [Ross, 2013]. Used to detect changes in the parameter of an Exponentially distributed sequence.
              # • GLRAdjusted and ExponentialAdjusted: Identical to the GLR and Exponential statistics, except with the finite-sample correction discussed in [Ross, 2013] which can lead to more powerful change detection.
              # • FET: Fishers Exact Test statistic, as in [Ross and Adams, 2012b]. Use to detect parameter changes in a Bernoulli sequence.
              # • Mann-Whitney: Mann-Whitney test statistic, as in [Ross et al, 2011]. Use to detect location shifts in a stream with a (possibly unknown) non-Gaussian distribution.
              # • Mood: Mood test statistic, as in [Ross et al, 2011]. Use to detect scale shifts in a stream with a (possibly unknown) non-Gaussian distribution.
              # • Lepage: Lepage test statistics in [Ross et al, 2011]. Use to detect location and/or shifts in a stream with a (possibly unknown) non-Gaussian distribution.
              # • Kolmogorov-Smirnov: Kolmogorov-Smirnov test statistic, as in [Ross et al 2012]. Use to detect arbitrary changes in a stream with a (possibly unknown) non-Gaussian distribution.
              # • Cramer-von-Mises: Cramer-von-Mises test statistic, as in [Ross et al 2012]. Use to detect arbitrary changes in a stream with a (possibly unknown) nonGaussian distribution.
    cpmType = ["Student", "GLR", "GLRAdjusted", "Mann-Whitney"] # IT DEPENDS ON THE PROBLEM!
    ## alpha (CPM1B): Significance level. By definition, the alpha level is the probability of rejecting the null hypothesis when it is true.
            # The allowable values for this argument are 0.05, 0.01, 0.005, 0.001
    alpha = [0.05, 0.01, 0.005, 0.001]
    ## ARL0: average number of observations before a false positive occurs, assuming that the sequence does not undergo a change. 
           # the package contains pre-computed values of the thresholds corresponding to several common values of the ARL0.
           # This means that only certain values for the ARL0 are allowed. 
           # the ARL0 must have one of the following values: 370, 500, 600, 700, ..., 1000, 2000, 3000, ..., 10000, 20000, ..., 50000.
    ARL0 = [370, 500, 600] # IT DEPENDS ON THE PROBLEM!
    ## startup : the number of observations after which monitoring begins. No change points will be flagged during this startup period.
               # This must be set to at least 20.
    startup = [20]
    ###
    
    
    ### SEGNEIGH: Segment Neighborhoods
    ## penalty: Choice of "None", "SIC", "BIC", "MBIC", "AIC", "Hannan-Quinn", "Asymptotic", "Manual" and "CROPS" penalties.
              # If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the
              # theoretical type I error is contained in the pen.value parameter. If CROPS is specified, the penalty range is 
              # contained in the pen.value parameter; note this is a vector of length 2 which contains the minimum and maximum penalty value.
              # Note CROPS can only be used if the method is "PELT". The predefined penalties listed DO count the changepoint as a parameter, 
              # postfix a 0 e.g."SIC0" to NOT count the changepoint as a parameter.
              # RV: MBIC does not work for SegNeigh
    penalty = ["None", "SIC", "BIC", "AIC", "Hannan-Quinn"] # ["Asymptotic"] ["Manual"]
    ## pen_value: The theoretical type I error e.g.0.05 when using the Asymptotic penalty.
                # A vector of length 2 (min,max) if using the CROPS penalty. 
                # The value of the penalty when using the Manual penalty option - this can be a numeric value or text giving the formula to use. 
                # Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, 
                # diffparam=difference in number of alternatve and null parameters.
    pen_value = [0] # [0.01, 0.05] np.logspace(-3, 3, num=101, endpoint=True, base=10.0)
    ## Q: The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Q = [3, 4, 5] # IT HIGHLY DEPENDS OF THE PROBLEM
    ## test_stat: The assumed test statistic / distribution of the data. Currently only "Normal" and "CUSUM" supported.
    test_stat = ["Normal", "CUSUM"]
    
    param_grid_SegNeigh = {'penalty':penalty, 'pen_value':pen_value, 'Q':Q, 'test_stat':test_stat}
    ###
    
    
    
    # print('TEST of CPDGridSearch class and values')
    # MyEstimator = CPDEstimator(algo='sbs')
    # grid_search = CPDGridSearch(estimator=MyEstimator, param_grid=param_grid_wbs)
    # grid_search.fit(data, trueCPs, delta)
    # grid_search_results = pd.DataFrame(data=grid_search.results_values_, columns=grid_search.results_names_)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    ###############################################################################
    