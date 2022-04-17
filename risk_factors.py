def CreateRiskFactors(dfs, n_pcs, gran_names=None, threshold=0.5, SEED=1):e
    '''
    From datasets extract the risk factors
    INPUT:
        dfs: list of datasets at different time granularities
        n_pcs: number of Principal Components to consider
        gran_names: list of strings containing the name of each granularity,
            same dimension of dfs
        threshold: float, correlation threshold for the multicollinearity filter
        SEED: int, seed to set random values
    OUTPUT:
        RFs: dataset containing the extracted risk factors
    '''
    from sklearn.decomposition import PCA
    from numpy import corrcoef
    from pandas import DataFrame

    if gran_names == None:
        gran_names = list()
        for df in range(len(dfs)): gran_names.append(f'{df}')
        print(gran_names)

    #----- Compute Principal Components
    PCs = list()
    for df, name in zip(dfs, gran_names):
        pca_temp = PCA(n_components=n_pcs, random_state=SEED)
        pca_temp.fit(df.values)
        PCs.append(DataFrame(pca_temp.transform(dfs[0].values),
                             index=dfs[0].index,
                             columns=[f'{name}_{k}th' \
                                      for k in range(1,n_pcs+1)]))
    
    #----- Apply multicollinearity filter to obtain the final Risk Factors
    RFs = PCs[0]
    for PC in PCs:
        for n, colM in enumerate(PC.columns):
            corr = 0
            for colD in RFs.columns:
                temp = abs(corrcoef(RFs[colD].values, PC[colM].values)[0,1])
                if temp > corr:
                    corr = temp
            if corr < 0.5:
                RFs[colM] = PC[colM]

    return RFs
