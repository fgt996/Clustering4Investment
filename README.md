# Clustering4Investment
This project starts from the research articles "An unsupervised learning framework for marketneutral portfolio" and ""Statistical arbitrage in the stock markets by the means of multiple time horizons clustering"". It contains some code related to the experimental part of the paper. In particular, a study of multiple time-horizons risk factors is carried out.

These are the starting points:

1) X: DataFrame, containing the considered assets universe in the training set. Each row represents a datetime, while each column represents the value of an asset.
2) X_Test: DataFrame, containing the considered assets universe in the test set. It is only exploited for comparison with benchmark strategies.
3) n_pcs: int, number of principal components extracted for each time horizon.
4) len_fold: int, length of each fold in the cross-validation.
5) n_win: int, number of windows to use in the pooled regression.


This is the scheme followed:

1) Extract risk factors via multiple time-horizons Principal Component Analysis, by exploiting a multicollinearity filter to avoid collinearity among risk factors.
2) Perform A-Lasso in order to understand, asset by asset, the subset of risk factors that significantly affect the considered asset.
3) Clusters the assets according to the exposition on the same risk factors: assets affected by the same risk factors are grouped together.
4) Create a market-neutral portfolio within each cluster: the weights are optimized to maximize the expected Sharpe Ratio of the resulting portfolio. Two types of constraints are set: Linear (the portfolio exposition on each risk factor is nil); Non-Linear (the total exposition, on both long and short positions, is unitary).
5) Among all the possible market-neutral portfolios (one within each cluster), select and create an equally-weighted portfolio made up of the optimal three ones.












## Reference:
"An unsupervised learning framework for marketneutral portfolio"
S.Cuomo, F.Gatta, F.Giampaolo, C.Iorio, F.Piccialli
Expert Systems with applications (2022) 
https://doi.org/10.1016/j.eswa.2021.116308


"Statistical arbitrage in the stock markets by the means of multiple time horizons clustering"
F.Gatta, C.Iorio, D.Chiaro, F.Giampaolo, S.Cuomo
Neural Computing and Applications (2023) 
https://link.springer.com/article/10.1007/s00521-023-08313-6
