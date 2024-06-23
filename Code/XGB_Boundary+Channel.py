import numpy as np
import pickle
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV 
from Process_Channel5200 import X_train, y_train
from Process_Channel5200 import X_test, y_test, tau_DNS_test, dudy_DNS_test, k_DNS_test, \
    c0_DNS_test, c2_DNS_test, uu_DNS, yplus_DNS, yplus_DNS_test, vv_DNS, ww_DNS, c, c0_DNS_train, \
    c2_DNS_train

#Inputparametrar för modellen med olika värden
param_grid = {
    'n_estimators': [ 250 , 400, 500],
    'gamma': [0 , 0.1, 0.2 ],
    'min_child_weight': [0.001 , 0.01, 0.1 ],
    'max_depth': [  1 , 2, 4 ],
    'learning_rate': [ 0.01, 0.1 , 0.2],
    'subsample': [0.2 , 0.5 , 0.7 ] 
}
def main():
#Evaluerar och skapar ny model med optimala värden samt kör denna på hela settet
    model_c_0 = XGBRegressor()
    grid_search = GridSearchCV(model_c_0, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c0_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c0: ", grid_search.best_score_)

    model_c_0 = XGBRegressor(**grid_search.best_params_)
    model_c_0.fit(X_train, c0_DNS_train)

    model_c_2 = XGBRegressor()
    grid_search = GridSearchCV(model_c_2, param_grid, cv = 5, scoring= 'neg_mean_absolute_percentage_error', verbose = 1)
    grid_search.fit(X_train, c2_DNS_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score MAPE c2: ", grid_search.best_score_)

    model_c_2 = XGBRegressor(**grid_search.best_params_)
    model_c_2.fit(X_train, c2_DNS_train)

    c0_XGB = model_c_0.predict(X_test)

    c2_XGB = model_c_2.predict(X_test)

    Results(c0_XGB, c2_XGB)

    
#Sparar c0 och c2 i separata filer på hårddisken 
    with open('XGB_c0.pkl', 'wb') as f:
        pickle.dump(model_c_0, f)

    with open('XGB_c2.pkl', 'wb') as f:
        pickle.dump(model_c_2, f)


def Results(c0_XGB, c2_XGB):

    c0_std=np.std(c0_XGB-c0_DNS_test)/(np.mean(c0_XGB.flatten()**2))**0.5
    c2_std=np.std(c2_XGB-c2_DNS_test)/(np.mean(c2_XGB.flatten()**2))**0.5
    c0_RMSE=np.sqrt(sum((c0_XGB-c0_DNS_test)**2)/len(c0_XGB))
    c2_RMSE=np.sqrt(sum((c2_XGB-c2_DNS_test)**2)/len(c2_XGB))
    c0_RMSPE=np.sqrt(sum((c0_XGB-c0_DNS_test)/(c0_DNS_test*len(c0_XGB)))**2)
    c2_RMSPE=np.sqrt(sum((c2_XGB-c2_DNS_test)/(c2_DNS_test*len(c2_XGB)))**2)

    a_11 = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_XGB + 6 * c2_XGB)
    uu_XGB = (a_11 + 0.6666) * k_DNS_test

    a_22 = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_XGB - 6 * c2_XGB)
    vv_XGB = (a_22 + 0.6666) * k_DNS_test

    a_33 = -1 / 6 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * c0_XGB
    ww_XGB = (a_33 + 0.6666) * k_DNS_test

    print('\nc0_error_RMSE',c0_RMSE)
    print('\nc2_error_RMSE',c2_RMSE)
    print('\nc0_error_RMSPE',c0_RMSPE)
    print('\nc2_error_RMSPE',c2_RMSPE)
    print('\nc0_error_std',c0_std)
    print('\nc2_error_std',c2_std)

        
    # plot coefficients XGB vs DNS
    fig,axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    axs[1].scatter(c0_XGB, yplus_DNS_test, marker="o", s=10, c="orangered", label="XGB")
    axs[1].plot(c[0,:], yplus_DNS,'b-', c="deepskyblue", label="DNS-data")
    axs[1].legend(loc="best", fontsize=12)
    axs[1].set_xlabel("$c0_XGB$", fontsize=14)
    fig.suptitle("Prediktioner för koefficienterna $c_i$ med XGB", fontsize=18)

    axs[0].scatter(c2_XGB, yplus_DNS_test, marker="o", s=10, c="yellowgreen", label="XGB")
    axs[0].plot(c[1,:], yplus_DNS,'b-', c="slateblue", label="DNS-data")
    axs[0].legend(loc="best", fontsize=12)
    axs[0].set_xlabel("$c2_XGB$", fontsize=14)
    axs[0].set_ylabel("$y^+$", fontsize=14)
    plt.show()


    # plot turbulence XGB vs DNS
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    ax[0].scatter(uu_XGB, yplus_DNS_test, marker="o", s=10, c="violet", label="XGB")
    ax[0].plot(uu_DNS, yplus_DNS, 'b-', c="slateblue", label="DNS-data")
    ax[0].legend(loc="best", fontsize=12)
    ax[0].set_xlabel("$\overline{u'u'}^+$", fontsize=14)
    ax[0].set_ylabel("$y^+$", fontsize=14)
    fig.suptitle("Prediktioner för alla stress tensorer med XGB", fontsize=18)

    ax[1].scatter(vv_XGB, yplus_DNS_test, marker="o", s=10, c="hotpink", label="XGB")
    ax[1].plot(vv_DNS, yplus_DNS, 'b-', c="slateblue", label="DNS-data")
    ax[1].set_xlabel("$\overline{v'v'}^+$", fontsize=14)
    ax[1].legend(loc="best", fontsize=12)

    ax[2].scatter(ww_XGB, yplus_DNS_test, marker="o", s=10, c="lightgreen", label="XGB")
    ax[2].plot(ww_DNS, yplus_DNS, 'b-', c="slateblue", label="DNS-data")
    ax[2].set_xlabel("$\overline{w'w'}^+$", fontsize=14)
    ax[2].legend(loc="best", fontsize=12)

    plt.show()

#Results(c0_XGB, c2_XGB)

if __name__ == '__main__':
    main()