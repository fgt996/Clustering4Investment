from pandas import Series

def Percentage_Profit(X):
    '''
    Return investment strategy profit
    INPUT:
        X: series, strategy to evaluate
    OUTPUT:
        res: int, profit
    '''
    res = 100*((X[-1]/X[0])-1)
    return res

def Profit_Factor(X):
    '''
    Return investment strategy profit factor
    INPUT:
        X: series, strategy to evaluate
    OUTPUT:
        res: int, profit factor
    '''
    up = 0
    down = 0
    for t in range(1,len(X)):
        diff = X[t]-X[t-1]
        if diff > 0:
            up = up + diff
        else:
            down = down - diff
    res = up/down
    return res

def Percentage_Drawdown(X):
    '''
    Return investment strategy percentage drawdown
    INPUT:
        X: series, strategy to evaluate
    OUTPUT:
        res: int, percentage drawdown
    '''
    draw_p = 0
    draw = 0
    old_max = X[0]
    for t in range(1,len(X)):
        if X[t] > old_max:
            old_max = X[t]
        else:
            diff = old_max - X[t]
            if diff > draw:
                draw = diff
                draw_p = diff/old_max
    res = 100*draw_p
    return res

def Recovery_Factor(X):
    '''
    Return investment strategy recovery factor
    INPUT:
        X: series, strategy to evaluate
    OUTPUT:
        res: int, recovery factor
    '''
    total_profit = X[-1] - X[0]
    draw = 0
    old_max = X[0]
    for t in range(1,len(X)):
        if X[t] > old_max:
            old_max = X[t]
        else:
            diff = old_max - X[t]
            if diff > draw:
                draw = diff
    res = total_profit / draw
    return res

def Sharpe_Ratio(X):
    '''
    Return investment strategy sharpe ratio
    INPUT:
        X: series, strategy to evaluate
    OUTPUT:
        res: int, sharpe ratio
    '''
    from pandas import Series
    from numpy import sqrt, std
    
    X_rend = Series(X).pct_change()[1:]
    mu = ( X[-1]/X[0] )**(252/len(X)) - 1
    sigma = sqrt(252)*std(X_rend)
    res = mu / sigma
    return res

def Sortino_Ratio(X):
    '''
    Return investment strategy sortino ratio
    INPUT:
        X: series, strategy to evaluate
    OUTPUT:
        res: int, sortino ratio
    '''
    from numpy import sqrt
    from pandas import Series
    
    X_rend = Series(X).pct_change()[1:]
    downside_risk = X_rend[X_rend < X_rend.mean()].std()
    mu = ( X[-1]/X[0] )**(252/len(X)) - 1
    sigmaD = sqrt(252)*downside_risk
    res = mu / sigmaD
    return res


def Strategy_Evaluation(X, verbose=True, out=False, n_round=3):
    '''
    Evaluate investment strategy according to well-known metrics
    INPUT:
        X: series, strategy to evaluate
        verbose: bool, manages the verbosity. Default=True
        out: bool, if True, return metrics dict as output. Default=False
        n_round: int, number of decimals considered (if <=0, no round).Default=3
    OUTPUT:
        res: dict, metrics of the strategy
    '''
    from numpy import round
    Properties = {'Percentage Profit': Percentage_Profit(X),
            'Profit Factor': Profit_Factor(X),
            'Percentage Drawdown': Percentage_Drawdown(X),
            'Recovery Factor': Recovery_Factor(X),
            'Sharpe Ratio': Sharpe_Ratio(X),
            'Sortino Ratio': Sortino_Ratio(X)}
    if verbose:
        for prop in Properties.keys():
            if prop.split(' ')[0] == 'Percentage':
                print(f'{prop}: {round(Properties[prop], n_round)}%')
            else:
                print(f'{prop}: {round(Properties[prop], n_round)}')
    if out:
        return Properties
