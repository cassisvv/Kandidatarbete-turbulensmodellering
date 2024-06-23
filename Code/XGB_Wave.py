import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV 
from Process_Wave import X_train
from Process_Wave import X_test, c1_DNS_train, c2_DNS_train, c3_DNS_train,  uu_DNS_train, vv_DNS_train, uv_DNS_train, ww_DNS_train

#Inputparametrar för modellen med olika värden
param_grid = {
    'n_estimators': [ 500, 10000],
    'gamma': [0 , 0.01 ],
    'min_child_weight': [0.001 ],
    'max_depth': [ 6, 15],
    'learning_rate': [ 0.3, 0.4],
    'subsample': [ 0.7, 0.9 ] 
    }

def main():
#Evaluerar och skapar ny model med optimala värden samt kör denna på hela settet
    model_c_1 = XGBRegressor()
    grid_search = GridSearchCV(model_c_1, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c1_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c1: ", grid_search.best_score_)

    model_c_1 = XGBRegressor(**grid_search.best_params_)
    model_c_1.fit(X_train, c1_DNS_train)

    model_c_2 = XGBRegressor()
    grid_search = GridSearchCV(model_c_2, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c2_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c2: ", grid_search.best_score_)

    model_c_2 = XGBRegressor(**grid_search.best_params_)
    model_c_2.fit(X_train, c2_DNS_train)

    model_c_3 = XGBRegressor()
    grid_search = GridSearchCV(model_c_3, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c3_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c3: ", grid_search.best_score_)

    model_c_3 = XGBRegressor(**grid_search.best_params_)
    model_c_3.fit(X_train, c3_DNS_train)

    model_uu = XGBRegressor()    
    grid_search = GridSearchCV(model_uu, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, uu_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE uu: ", grid_search.best_score_)

    model_uu = XGBRegressor(**grid_search.best_params_)
    model_uu.fit(X_train, uu_DNS_train)

    model_vv = XGBRegressor()    
    grid_search = GridSearchCV(model_vv, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, vv_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE vv: ", grid_search.best_score_)

    model_vv = XGBRegressor(**grid_search.best_params_)
    model_vv.fit(X_train, vv_DNS_train)

    model_ww = XGBRegressor()    
    grid_search = GridSearchCV(model_ww, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, ww_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE ww: ", grid_search.best_score_)

    model_ww = XGBRegressor(**grid_search.best_params_)
    model_ww.fit(X_train, ww_DNS_train)

    model_uv = XGBRegressor()    
    grid_search = GridSearchCV(model_uv, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, uv_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE uv: ", grid_search.best_score_)

    model_uv = XGBRegressor(**grid_search.best_params_)
    model_uv.fit(X_train, uv_DNS_train)


 
 

#Sparar c1, c2 och c3 i separata filer på hårddisken 
    with open('XGB_2D_c1.pkl', 'wb') as f:
        pickle.dump(model_c_1, f)

    with open('XGB_2D_c2.pkl', 'wb') as f:
        pickle.dump(model_c_2, f)
    
    with open('XGB_2D_c3.pkl', 'wb') as f:
        pickle.dump(model_c_3, f)
   
    with open('XGB_2D_uu.pkl', 'wb') as f:
        pickle.dump(model_uu, f)

    with open('XGB_2D_vv.pkl', 'wb') as f:
        pickle.dump(model_vv, f)

    with open('XGB_2D_ww.pkl', 'wb') as f:
        pickle.dump(model_ww, f)

    with open('XGB_2D_uv.pkl', 'wb') as f:
        pickle.dump(model_uv, f)

if __name__ == '__main__':
    main()

