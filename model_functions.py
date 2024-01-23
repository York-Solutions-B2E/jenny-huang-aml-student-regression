import pandas as pd 
import numpy as np
from sklearn.metrics import mutual_info_score, mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression, SelectKBest, SelectFromModel
from sklearn.model_selection import GridSearchCV


#run a model and get evaluation metrics
def get_model_stats (model, x_train, x_test, y_train, y_test): 
    model.fit(x_train, y_train)
    pred = np.round(model.predict(x_test))
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    #a proxy for percent error- need to fix
    mae = mean_absolute_error(y_test, pred)
    relative_e = mae/y_test.mean()
    return([r2, mse, mae, relative_e])

#run a model for various different selection techniques and return eval metrics
def test_features(model, x_train, x_test, y_train, y_test, selectors): 
    stats = pd.DataFrame(columns = ['selector', 'features', 'r2', 'mse', 'mae', 'relative error'])
    for selector in selectors:
        selector.fit(x_train, y_train)
        features = list(selector.get_feature_names_out())
        train_temp = x_train[features]
        test_temp = x_test[features]
        m_stats = get_model_stats(model, train_temp, test_temp, y_train, y_test)
        stats.loc[len(stats.index)]= [str(selector), features] + m_stats
    return(stats)

def compare_models(models, x_train, x_test, y_train, y_test, selectors, metric):
    stats = pd.DataFrame(columns = ['model', 'k features', 'features', 'r2', 'mse', 'mae', 'relative error'])
    for m in models:
        temp = test_features(m, x_train, x_test, y_train, y_test, selectors)
        if metric == 'r2': 
            best = [str(m)] + temp[temp[metric] == temp[metric].max()].values.tolist()[0]
        else: 
            best = [str(m)] + temp[temp[metric] == temp[metric].min()].values.tolist()[0]
        best[1] = len(best[2])
        stats.loc[len(stats.index)] = best
    stats = stats.sort_values(by = metric).reset_index(drop=True)
    return stats 

def tune_model(model, x, y_train, params):
    tuner = GridSearchCV(estimator=model, param_grid = params, scoring= 'neg_mean_absolute_error')
    tuner.fit(x,y_train)
    return pd.DataFrame(tuner.cv_results_).sort_values(by='rank_test_score')[['params', 'mean_test_score']]