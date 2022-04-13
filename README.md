# Clustering4Investment
This project starts from the research article "An unsupervised learning framework for marketneutral portfolio". It contains some code related to the experimental part of the paper. In particular, a study of multiple time-horizons risk factors is carried out.

These are the starting points:

1) X: DataFrame, containing the considered assets universe. Each row represents a datetime, while each column represents the value of an asset.
2) test_start: str or datetime, datetime < test_start are assumed to be in the training set, while datetime >= test_start are assumed to be in the test set (it is optional, in that the code can be used to simply create an optimal portfolio for future investments, without evaluating it).


This is the scheme followed:

1) Extract risk factors via multiple time-horizons Principal Component Analysis, by exploiting a multicollinearity filter to avoid collinearity among risk factors.
2) Perform A-Lasso in order to understand, asset by asset, the subset of risk factors that significantly affect the considered asset.
3) Clusters the assets according to the exposition on the same risk factors: assets affected by the same risk factors are grouped together.
4) Create a market-neutral portfolio within each cluster: the weights are optimized to maximize the expected Sharpe Ratio of the resulting portfolio. Two types of constraints are set: Linear (the portfolio exposition on each risk factor is nil); Non-Linear (the total exposition, on both long and short positions, is unitary).
5) Among all the possible market-neutral portfolios (one within each cluster), select and create an equally-weighted portfolio made up of the optimal three ones.












## Reference:
### An unsupervised learning framework for marketneutral portfolio
S.Cuomo, F.Gatta, F.Giampaolo, C.Iorio, F.Piccialli
Expert Systems with applications (2022) 
https://doi.org/10.1016/j.eswa.2021.116308
