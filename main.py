from C4I import benchmarks, clustering, evaluation, investment, risk_factors, utils
import pandas as pd
import numpy as np
import pickle


def An_Unsupervised_Learning_Framework(X, n_pcs, len_fold, n_win,temp_path=None,
                                scaler_pre='MinMax', coll_threshold=0.5, SEED=1,
                                lasso_fold=5, min_rfs=2, max_rfs=4, n_round=3,
                                scaler='Standard',
                                opt_hyper_port={'maxiter':200, 'disp':True},
                                port_select=3, verbose=True):
    '''
    INPUT:
        X: DataFrame, contains assets daily prices
        n_pcs: number of Principal Components to consider
        len_fold: int, length of each fold
        n_win: int, number of windows for the pooled regression
        temp_path: None or str, path for temp savings. If None, there are no
            temp savings. Default=None
        scaler_pre: str representing the scaler to use in preprocessing, either
            'Standard' or 'MinMax'. Other values will result in no scaling.
            Default='Standard'
        coll_threshold: float, correlation threshold for collinearity filter
        SEED: int, seed to set random values
        lasso_fold: int, number of folds for the cross validation. Default = 5
        min_rfs: int, minimum number of risk factors to be saved. Default = 2
        max_rfs: int, maximum number of risk factors to be saved. Default = 4
        n_round: int, number of decimals considered (if <=0, no round).Default=3
        scaler: str representing the scaler to use, either 'Standard' or
            'MinMax'. Other values will result in no scaling. Default='Standard'
        opt_hyper_port: dcit, hyperparameters for the minimizer, containing two
            keys: 'maxiter' and 'disp'. Default={'maxiter':200, 'disp':True}
        port_select: int, number of portfolios to select. Default=3
        verbose: bool, manages the verbosity. Default=True
    OUTPUT:
        target_portfolios: list, containing weights of the optimal portfolio
    '''
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    #----- From prices to returns
    X.index = pd.to_datetime(X.index)
    # Monthly returns
    XM = utils.DailyPrice2MonthlyReturn(X)
    # Daily returns
    XD = utils.DailyPrice2DailyReturn(X)
    # Scale data
    if scaler_pre == 'Standard':
        XD = pd.DataFrame( StandardScaler().fit_transform(XD.values),
                          index = XD.index, columns = XD.columns )
        XM = pd.DataFrame( StandardScaler().fit_transform(XM.values),
                          index = XM.index, columns = XM.columns )
    elif scaler_pre == 'MinMax':
        XD = pd.DataFrame( MinMaxScaler().fit_transform(XD.values),
                          index = XD.index, columns = XD.columns )
        XM = pd.DataFrame( MinMaxScaler().fit_transform(XM.values),
                          index = XM.index, columns = XM.columns )
    
    #----- Create Risk Factors
    if temp_path != None:
        try:
            PCs = pd.read_csv(f'{temp_path}/RiskFactors.csv', index_col=0)
            PCs.index = pd.to_datetime(PCs.index)
        except:
            PCs = risk_factors.CreateRiskFactors([XD, XM], n_pcs,
                                                 gran_names=['Daily','Monthly'],
                                                 threshold=coll_threshold,
                                                 SEED=SEED)
            PCs.to_csv(f'{temp_path}/RiskFactors.csv')
    else:
        PCs = risk_factors.CreateRiskFactors([XD, XM], n_pcs,
                                             gran_names=['Daily', 'Monthly'],
                                             threshold=coll_threshold,SEED=SEED)
    if verbose:
        print('Risk Factors successfully extracted!\n')

    #----- Apply Features Selection
    #Check if there are previous savings
    if temp_path != None:
        try:
            with open(f'{temp_path}/Saved_RFs.pickle','rb') as f:
                saved_rf = pickle.load(f)
        except:
            saved_rf = dict()
    else:
        saved_rf = dict()
    #Go on with features selection
    Grid = {'alpha':np.logspace(-11, -2, num=25, base=np.e),
        'tau':np.linspace(0.5, 1.5, 11)}
    for col in XD.columns[len(saved_rf.keys()):]:
        if verbose:
            print(f'Optimizing hyperparameters for asset {col}')
        flag, res = clustering.A_Lasso_Grid_Search(series=XD[col],
                                                   Exog=PCs.values,
                                                   grid=Grid,
                                                   n_fold=lasso_fold,
                                                   len_fold=len_fold,
                                                   min_rfs=min_rfs,
                                                   max_rfs=max_rfs,
                                                   scaler=scaler)
        #If the result is positive, store the hyperparameters
        if flag:
            saved_rf[col] = res
            if verbose:
                print(f'Optimization result: {res}')
        #Otherwise, make a new, more accurate search
        else:
            if res == -1:
                temp_Grid = {'alpha':np.logspace(-16, -11,num=25,base=np.e),
                            'tau':np.linspace(0.5, 1.5, 11)}
            elif res == 1:
                temp_Grid = {'alpha':np.logspace(-3, 3, num=25, base=np.e),
                            'tau':np.linspace(0.5, 1.5, 11)}
            else:
                temp_Grid = {'alpha':np.logspace(-11, -2,num=40,base=np.e),
                            'tau':np.linspace(0.5, 2, 20)}
            flag, res = clustering.A_Lasso_Grid_Search(series=XD[col],
                                                       Exog=PCs.values,
                                                       grid=temp_Grid,
                                                       n_fold=lasso_fold,
                                                       len_fold=len_fold,
                                                       min_rfs=min_rfs,
                                                       max_rfs=max_rfs,
                                                       scaler=scaler)
            if flag:
                saved_rf[col] = res
                if verbose:
                    print(f'Optimization result: {res}')
        if temp_path != None:
            with open(f'{temp_path}/Saved_RFs.pickle','wb') as f:
                pickle.dump(saved_rf, f)

    #----- Create Clustering
    #Create clusters
    clusters = clustering.Clustering(saved_rf)
    if verbose:
        print('Total clusters:')
        for n_clust, cluster in enumerate(clusters):
            print(f'Cluster {n_clust}, Risk factors: {cluster[0]},\
            Cluster Dimension: {len(cluster[1])}')
            print(f'Assets in the cluster: {cluster[1]}')
    #Save useful clusters
    clusters = clustering.UsefulClusters(clusters, saved_rf)
    if verbose:
        print('\nUseful clusters:')
        for n_clust, cluster in enumerate(clusters):
            print(f'Cluster {n_clust}, Risk factors: {cluster[0]}')
            print(f'Assets in the cluster: {cluster[1]}')

    #----- Investment Strategy
    target_portfolios = investment.Create_Portfolios(XD=XD, Exog=PCs.values,
                                        clusters=clusters,
                                        target_function=utils.Expected_Sharpe,
                                        windows_number=n_win,
                                        scaler=scaler, opt_hyper=opt_hyper_port,
                                        port_to_select=port_select)
    #Adjust portfolios weight to obtain one output portfolio
    for port in target_portfolios:
        port[1] = port[1]/port_select
    return target_portfolios

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = 'Dataset'
    X = pd.read_csv(f'{path}/Stock_Market.csv', index_col=0)
    X.index = pd.to_datetime(X.index)

    test_start = '2019-12-21'
    XD = utils.DailyPrice2DailyReturn(X)
    X_Test = XD[ XD.index >= test_start ]
    X = X[ X.index < test_start ]
    
    #Obtain portfolio value in the test set
    target_portfolio = An_Unsupervised_Learning_Framework(X, n_pcs=7,
                                                          len_fold=42, n_win=50)
    
    #Evaluate results
    port_test = investment.Investment(X_Test, target_portfolio, initial_capital=1000)
    print('\nProposed strategy')
    evaluation.Strategy_Evaluation(port_test)
    print('\n')

    #Evaluate benchmarks
    min_test = benchmarks.Minimal_Variance_Portfolio(XD, X_Test)
    print('Minimal Variance Portfolio')
    evaluation.Strategy_Evaluation(min_test)
    print('\n')

    mean_var_test = benchmarks.Mean_Variance_Portfolio(XD, X_Test)
    print('Mean-Variance Portfolio')
    evaluation.Strategy_Evaluation(mean_var_test)
    print('\n')

    #Plot the results
    port_test.plot(figsize=(16,6))
    min_test.plot()
    mean_var_test.plot()
    plt.grid()
    plt.legend()
    plt.show()
