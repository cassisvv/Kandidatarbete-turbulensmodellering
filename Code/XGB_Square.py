import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV 
from Process_Square150 import X_train, \
         c_DNS_train_list


#Inputparametrar för modellen med olika värden
param_grid = {
    'n_estimators': [100, 1000, 10000 ],
    # 'gamma': [0 , 0.01, 0.001 ],
    # 'min_child_weight': [0.001 ],
    'max_depth': [ 8 , 25, 100],
    'learning_rate': [1e-5, 1e-3, 0.1],
    'subsample': [ 0.5, 0.7, 0.9 ] 
    }

def main():
#Evaluerar och skapar ny model med optimala värden samt kör denna på hela settet
    model_c_1 = XGBRegressor()
    grid_search = GridSearchCV(model_c_1, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c_DNS_train_list[:, 0])

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c1: ", grid_search.best_score_)

    model_c_1 = XGBRegressor(**grid_search.best_params_)
    model_c_1.fit(X_train, c_DNS_train_list[:,0])

    model_c_2 = XGBRegressor()
    grid_search = GridSearchCV(model_c_2, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c_DNS_train_list[:, 1])

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c2: ", grid_search.best_score_)

    model_c_2 = XGBRegressor(**grid_search.best_params_)
    model_c_2.fit(X_train, c_DNS_train_list[:, 1])

    model_c_3 = XGBRegressor()
    grid_search = GridSearchCV(model_c_3, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c_DNS_train_list[:,2])

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c3: ", grid_search.best_score_)

    model_c_3 = XGBRegressor(**grid_search.best_params_)
    model_c_3.fit(X_train, c_DNS_train_list[ :,2])

    model_c4 = XGBRegressor()    
    grid_search = GridSearchCV(model_c4, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c_DNS_train_list[:, 3])

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c4: ", grid_search.best_score_)

    model_c4 = XGBRegressor(**grid_search.best_params_)
    model_c4.fit(X_train, c_DNS_train_list[:, 3])

    model_c5 = XGBRegressor()    
    grid_search = GridSearchCV(model_c5, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c_DNS_train_list[:, 4])

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c5: ", grid_search.best_score_)

    model_c5 = XGBRegressor(**grid_search.best_params_)
    model_c5.fit(X_train, c_DNS_train_list[:, 4])

 
#Sparar c1, c2, c3, c4 och c5 i separata filer på hårddisken 
    with open('XGB_SqD_c1.pkl', 'wb') as f:
        pickle.dump(model_c_1, f)

    with open('XGB_SqD_c2.pkl', 'wb') as f:
        pickle.dump(model_c_2, f)
    
    with open('XGB_SqD_c3.pkl', 'wb') as f:
        pickle.dump(model_c_3, f)
   
    with open('XGB_SqD_c4.pkl', 'wb') as f:
        pickle.dump(model_c4, f)

    with open('XGB_SqD_c5.pkl', 'wb') as f:
        pickle.dump(model_c5, f)

if __name__ == '__main__':
    main()





