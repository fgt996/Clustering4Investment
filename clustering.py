def Clustering(saved_rf):h
    '''
    Create the clusters from Idx saved for each asset
    INPUT:
    saved_rf: dict, containing saved risk factors of each asset
    OUTPUT
    cluster: list of couple (idx, risk factors) representing each cluster
    '''
    #Create empty clusters set
    clusters = []
    for asset in saved_rf.keys():
        flag = True
        #Check if asset belongs to an existing cluster
        for cluster in clusters:
            if cluster[0] == saved_rf[asset]:
                cluster[1].append( asset )
                flag = False
        #Otherwise, create a new cluster
        if flag:
            clusters.append( [saved_rf[asset], [asset]] )
    return clusters


def UsefulClusters(clusters, saved_rf):
    '''
    Discard usefulness clusters
    INPUT:
    clusters: list, containing couple (idx, risk factors) for each cluster
    saved_rf: dict, containing saved risk factors and cluster for each asset
    OUTPUT
    to_save: list, saved clusters
    '''
    to_save = list()
    for cluster in clusters:
        if len(cluster[1]) > len(cluster[0]):
            to_save.append( cluster )
    return to_save
  
  
def A_Lasso_Grid_Search(series, Exog, grid, n_fold=5, len_fold=21, min_rfs=2,
                        max_rfs=4, scaler='Standard'):
    '''
    From datasets extract the risk factors
    INPUT:
        series: series, asset return.
        Exog: array, numpy 2d array containing features.
        grid: dict, containing two keys, 'alpha' and 'tau', and their grid of
            admissible values
        n_fold: int, number of folds for the cross validation. Default = 5
        len_fold: int, length of each fold. Default = 21
        min_rfs: int, minimum number of risk factors to be saved. Default = 2
        max_rfs: int, maximum number of risk factors to be saved. Default = 4
        scaler: str representing the scaler to use, either 'Standard' or
            'MinMax'. Other values will result in no scaling. Default='Standard'
    OUTPUT:n_pcs
        flag: bool, if True optimization has been successful
        saved_rf: list containing saved risk factors.
        error: int, error code. -1 if too few risk factors are saved;
            1 if too many risk factors are saved; 0 unknown

    '''
    from numpy import array, concatenate, dot, diag, inf
    from statsmodels.regression.linear_model import OLS
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    #----- Preprocessing
    #Define working variables
    best_score = inf
    flag = False
    tot = 0

    #----- Grid Search
    for alpha in grid['alpha']:
        for tau in grid['tau']:
            #Define input and output
            if scaler == 'Standard':
                scalerX = StandardScaler().fit(Exog)
                XTrain = scalerX.transform(Exog)
                scalerY = StandardScaler().fit(series.values.reshape(-1,1))
                YTrain = scalerY.transform(series.values.reshape(-1,1))
            elif scaler == 'MinMax':
                scalerX = MinMaxScaler().fit(Exog)
                XTrain = scalerX.transform(Exog)
                scalerY = MinMaxScaler().fit(series.values.reshape(-1,1))
                YTrain = scalerY.transform(series.values.reshape(-1,1))
            else:
                XTrain = Exog
                YTrain = series.values.reshape(-1,1)
            #----- Apply A-LASSO to understand if the considered hyperparameters
                #save an admissible number of features
            #Define and Fit the OLS to obtain A-LASSO weights and rescale
                #input matrix accordingly
            mdl_ols = OLS(YTrain, XTrain).fit()
            XTrain = dot(XTrain,
                            diag([abs(k)**tau for k in mdl_ols.params]))
            #Apply A-LASSO and verify the number of saved features
            mdl_a_lasso = OLS(YTrain, XTrain).fit_regularized(L1_wt=1,
                                                              alpha=alpha)
            count = 0
            indices = []
            for idx in range(Exog.shape[1]):
                if abs(mdl_a_lasso.params[idx]) > 0:
                    count += 1; tot += 1
                    indices.append(idx)
            #if the number of saved features is admissible, go on
            if  ( (count >= min_rfs) and (count <= max_rfs) ):
                flag = True
                total_loss = 0
                for fold in range(n_fold):
                  #Split train and test according to Cross Validation
                    XTrain = Exog[:-(fold+1)*len_fold]
                    YTrain = series.values[:-(fold+1)*len_fold]
                    YTrain = YTrain.reshape(-1,1)
                    if fold == 0:
                        XTest = Exog[-(fold+1)*len_fold:]
                        YTest = series.values[-(fold+1)*len_fold:]
                    else:
                        XTest = Exog[-(fold+1)*len_fold:-fold*len_fold]
                        YTest = series.values[-(fold+1)*len_fold:-fold*len_fold]

                    #Scale the data
                    if scaler == 'Standard':
                        scalerX = StandardScaler().fit(XTrain)
                        XTrain = scalerX.transform(XTrain)
                        XTest = scalerX.transform(XTest)
                        scalerY = StandardScaler().fit(YTrain)
                        YTrain = scalerY.transform(YTrain)
                    elif scaler == 'MinMax':
                        scalerX = MinMaxScaler().fit(XTrain)
                        XTrain = scalerX.transform(XTrain)
                        XTest = scalerX.transform(XTest)
                        scalerY = MinMaxScaler().fit(YTrain)
                        YTrain = scalerY.transform(YTrain)
                    
                    #Define and Fit the OLS to obtain A-LASSO weights
                    mdl_ols = OLS(YTrain, XTrain).fit()
                    XTrain = dot(XTrain,
                                    diag([abs(k)**tau for\
                                             k in mdl_ols.params]))
                    #Apply A-LASSO and verify the number of saved features
                    mdl_a_lasso = OLS(YTrain,
                                      XTrain).fit_regularized(L1_wt=1,
                                                              alpha=alpha)
                    FinalTrain = array([1]*len(YTrain), ndmin=2).T
                    FinalTest = array([1]*len(YTest), ndmin=2).T
                    for idx in range(Exog.shape[1]):
                        if abs(mdl_a_lasso.params[idx]) > 0:
                            FinalTrain = concatenate([FinalTrain,
                                                         XTrain.T[idx:idx+1].T],
                                                        axis=1)
                            FinalTest = concatenate([FinalTest,
                                                        XTest.T[idx:idx+1].T],
                                                       axis=1)
                    #Once the features selection is performed, apply OLS to
                        #compute regression and obtain the mse
                    mdl_ols = OLS(YTrain, FinalTrain).fit()
                    pred = mdl_ols.predict(FinalTest).reshape(-1,1)
                    pred = scalerY.inverse_transform(pred)
                    total_loss += mse(YTest, pred) / n_fold
                if total_loss < best_score:
                    best_score = total_loss
                    top_alpha = alpha
                    top_tau = tau
                    saved_rf = indices
    
    #Assess if at least one hyperparameters couple saves an admissible
        #number of features
    if flag:
        return flag, saved_rf
    else:
        #Understand what is the problem
        if tot <= (min_rfs-1) * (len(grid['alpha']) * len(grid['tau'])):
            error = -1
        elif tot >= (max_rfs+1) * (len(grid['alpha']) * len(grid['tau'])):
            error = 1
        else:
            error = 0
        return flag, error
