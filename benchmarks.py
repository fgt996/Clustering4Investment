from utils import l1_norm, Return2Value, Var_Port, Mean_Variance

p
def Minimal_Variance_Portfolio(X_Train, X_Test,
                               opt_hyp={'maxiter':200,'disp':True,'ftol':1e-12},
                               initial_capital=1000):
    '''
    Compute values for Minimal Variance Portfolio benchmark
    INPUT:
    	X_Train: DataFrame, daily returns for the training set
    	X_Test: DataFrame, daily returns for the test set
        opt_hyp: dcit, hyperparameters for the minimizer, containing three
            keys: 'maxiter', 'disp' and 'ftol'.
            Default={'maxiter':200, 'disp':True 'ftol':1e-12}
        initial_capital: int, initial value fo the strategy. Default=1000
    OUTPUT:
        port_minimal: Minimal Variance portfolio strategy results
    '''
    from pandas import Series
    from numpy import array, zeros
    from scipy.optimize import minimize, Bounds, NonlinearConstraint
    
    #Define nonlinear constraint and domain
    constr_non = NonlinearConstraint(l1_norm, 0.9999, 1.0001)
    dominio = Bounds(-1, 1)
    w0 = [1/X_Train.shape[1]] * X_Train.shape[1]
    #Minimize portfolio variance
    minimal_variance = minimize(Var_Port, w0,
                            args=(X_Train.values),
                            method="SLSQP",
                            options=opt_hyp,
                            constraints=[constr_non],
                            bounds=dominio)
    weights = minimal_variance.x
    #Create Minimal Variance returns
    port_minimal = zeros(len(X_Test.index))
    for n_col, col in enumerate(X_Test.columns):
        port_minimal += X_Test[col] * weights[n_col]
    #From returns to values
    port_minimal = Series(Return2Value(port_minimal, initial_capital),
                             index=X_Test.index, name='Minimal Variance')
    return port_minimal
    
    
def Mean_Variance_Portfolio(X_Train, X_Test,
                            opt_hyp={'maxiter':250,'disp':True,'ftol':1e-3},
                            initial_capital=1000):
    '''
    Compute values for Mean-Variance Portfolio benchmark
    INPUT:
    	X_Train: DataFrame, daily returns for the training set
    	X_Test: DataFrame, daily returns for the test set
        opt_hyp: dcit, hyperparameters for the minimizer, containing three
            keys: 'maxiter', 'disp' and 'ftol'.
            Default={'maxiter':200, 'disp':True 'ftol':1e-12}
        initial_capital: int, initial value fo the strategy. Default=1000
    OUTPUT:
        port_mean_var: Mean-Variance portfolio strategy results
    '''
    from pandas import Series
    from numpy import array, zeros
    from scipy.optimize import minimize, Bounds, NonlinearConstraint
    
    #Define nonlinear constraint and domain
    constr_non = NonlinearConstraint(l1_norm, 0.9999, 1.0001)
    dominio = Bounds(-1, 1)
    w0 = [1/X_Train.shape[1]] * X_Train.shape[1]
    #Minimize portfolio variance
    mean_variance = minimize(Mean_Variance, w0,
                            args=(X_Train.values),
                            method="SLSQP",
                            options=opt_hyp,
                            constraints=[constr_non],
                            bounds=dominio)
    weights = mean_variance.x
    #Create Minimal Variance returns
    port_mean_var = zeros(len(X_Test.index))
    for n_col, col in enumerate(X_Test.columns):
        port_mean_var += X_Test[col] * weights[n_col]
    #From returns to values
    port_mean_var = Series(Return2Value(port_mean_var, initial_capital),
                             index=X_Test.index, name='Mean-Variance')
    return port_mean_var


def Mean_Variance_Portfolio_OLD(X_Test,
                            opt_hyp={'maxiter':250,'disp':True,'ftol':1e-3},
                            initial_capital=1000):
    '''
    Compute values for Mean-Variance Portfolio benchmark
    INPUT:
        X_Test: DataFrame, daily series of index
        opt_hyp: dcit, hyperparameters for the minimizer, containing three
            keys: 'maxiter', 'disp' and 'ftol'.
            Default={'maxiter':200, 'disp':True 'ftol':1e-12}
        initial_capital: int, initial value fo the strategy. Default=1000
    OUTPUT:
        port_mean_var: Mean-Variance portfolio strategy results
    '''
    from pandas import Series
    from numpy import array, zeros
    from scipy.optimize import minimize, Bounds, NonlinearConstraint
    
    #Define nonlinear constraint and domain
    constr_non = NonlinearConstraint(l1_norm, 0.9999, 1.0001)
    dominio = Bounds(-1, 1)
    w0 = [1/X.shape[1]] * X.shape[1]
    #Minimize portfolio variance
    mean_variance = minimize(Mean_Variance, w0,
                            args=(X_Test.values),
                            method="SLSQP",
                            options=opt_hyp,
                            constraints=[constr_non],
                            bounds=dominio)
    weights = mean_variance.x
    #Create Minimal Variance returns
    port_mean_var = zeros(len(X_Test.index))
    for n_col, col in enumerate(X_Test.columns):
        port_mean_var += X_Test[col] * weights[n_col]
    #From returns to values
    port_mean_var = Series(Return2Value(port_mean_var, initial_capital),
                             index=X_Test.index, name='Mean-Variance')
    return port_mean_var
