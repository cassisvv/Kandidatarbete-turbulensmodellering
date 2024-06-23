from Process_Boundary11000 import X_test,y_test, tau_DNS_test, dudy_DNS_test, k_DNS_test, \
    c0_DNS_test, c2_DNS_test, uu_DNS_test, yplus_DNS, yplus_DNS_test, ww_DNS_test, vv_DNS_test, \
    X_test, y_test, yplus_DNS_test, c, uu_DNS, vv_DNS, ww_DNS
#from random_forest import c_RFR
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


# load XGB_model
XGB_model_c0 = pickle.load(open('XGB_c0_Boundary.pkl', 'rb'))
XGB_model_c2 = pickle.load(open('XGB_c2_Boundary.pkl', 'rb'))

# XGB prediction
c0_XGB = XGB_model_c0.predict(X_test)
c2_XGB = XGB_model_c2.predict(X_test)

# constants for Const model
c0_const = -0.05 + 0.21
c2_const = 0.11

# make tensors for NN
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
neural_net = torch.load('Boundary11000depth2width50epochs10000batchsize5rate0.5.pt')
preds = neural_net(X_test_tensor)

# NN prediction
c_NN = preds.detach().numpy()
c0_NN = c_NN[:, 0]
c2_NN = c_NN[:, 1]


# calculate aij and turbulence for XGB
a_11_XGB = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_XGB + 6 * c2_XGB)
uu_XGB = (a_11_XGB + 0.6666) * k_DNS_test

a_22_XGB = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_XGB - 6 * c2_XGB)
vv_XGB = (a_22_XGB + 0.6666) * k_DNS_test

a_33_XGB = -1 / 6 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * c0_XGB
ww_XGB = (a_33_XGB + 0.6666) * k_DNS_test

# calculate aij and turbulence for Const
a_11_const = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_const + 6 * c2_const)
uu_const = (a_11_const + 0.6666) * k_DNS_test

a_22_const = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_const - 6 * c2_const)
vv_const = (a_22_const + 0.6666) * k_DNS_test

a_33_const = -1 / 6 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * c0_const
ww_const = (a_33_const + 0.6666) * k_DNS_test

# calculate aij for NN, and stress tensors
a_11_NN = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_NN + 6 * c2_NN)
uu_NN = (a_11_NN + 0.6666) * k_DNS_test

a_22_NN = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_NN - 6 * c2_NN)
vv_NN = (a_22_NN + 0.6666) * k_DNS_test

a_33_NN = -1 / 6 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * c0_NN
ww_NN = (a_33_NN + 0.6666) * k_DNS_test

# calculate aij for DNS
a_11_DNS_test = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_DNS_test + 6 * c2_DNS_test)
a_22_DNS_test = 1 / 12 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * (c0_DNS_test - 6 * c2_DNS_test)
a_33_DNS_test = -1 / 6 * tau_DNS_test ** 2 * dudy_DNS_test ** 2 * c0_DNS_test



# calculate and print all errors

variables = ['uu', 'vv', 'ww', 'c0', 'c2']
models = ['NN' , 'XGB', 'const']
metrics = ['RMSPE', 'MAPE']

# Outer loop for variables
for var in variables:
    # Inner loop for models
    for model in models:
        # Load data for DNS and model output
        dns_data = locals()[f'{var}_DNS_test']
        model_data = locals()[f'{var}_{model}']

        # Calculate metrics
        for metric in metrics:
            if metric == 'RMSPE':
                value = np.sqrt(sum((dns_data - model_data)**2) / sum(dns_data**2))*100
            elif metric == 'MAPE':
                value = sum(np.abs(dns_data - model_data)) / sum(np.abs(dns_data))*100
            value = round(value, 2)
            
            # Print or use the results
            print(f'{var}_{metric}_{model}: {value}%')


# plot coefficients NN, RFR and XGB vs DNS
fig,axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
axs[1].plot(c[1,:], yplus_DNS,'b-', c="deepskyblue", label="DNS-data")
axs[1].scatter(c2_NN, yplus_DNS_test, marker="o", s=10, c="orangered", label="Neuralt nätverk")
axs[1].scatter(c2_XGB, yplus_DNS_test, marker="o", s=10, c="darkorchid", label="RFR")
axs[1].legend(loc="best", fontsize=12)
axs[1].set_xlabel("$c_2$", fontsize=14)
axs[0].set_ylabel("$y^+$", fontsize=14)

axs[0].plot(c[0,:], yplus_DNS,'b-', c="deepskyblue", label="DNS-data")
axs[0].scatter(c0_NN, yplus_DNS_test, marker="o", s=10, c="orangered", label="Neuralt nätverk")
axs[0].scatter(c0_XGB, yplus_DNS_test, marker="o", s=10, c="darkorchid", label="RFR")
axs[0].legend(loc="best", fontsize=12)
axs[0].set_xlabel("$c_0$", fontsize=14)
fig.suptitle("Prediktioner för koefficienterna $c_i$", fontsize=18)
plt.show()



# plot turbulence NN, RFR and XGB vs DNS
fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
ax[0].plot(uu_DNS, yplus_DNS, 'b-', c="deepskyblue", label="DNS-data")
ax[0].scatter(uu_NN, yplus_DNS_test, marker="o", s=10, c="orangered", label="Neuralt nätverk")
ax[0].scatter(uu_XGB, yplus_DNS_test, marker="o", s=10, c="darkorchid", label="RFR")
ax[0].scatter(uu_const, yplus_DNS_test, marker="o", s=10, c="orange", label="Konstant c")
ax[0].legend(loc="best", fontsize=12)
ax[0].set_xlabel("$\overline{u'u'}^+$", fontsize=14)
ax[0].set_ylabel("$y^+$", fontsize=14)
ax[0].set_xlim(0,10)
fig.suptitle("Prediktioner för alla stresstensorer", fontsize=18)

ax[1].plot(vv_DNS, yplus_DNS, 'b-', c="deepskyblue", label="DNS-data")
ax[1].scatter(vv_NN, yplus_DNS_test, marker="o", s=10, c="orangered", label="Neuralt nätverk")
ax[1].scatter(vv_XGB, yplus_DNS_test, marker="o", s=10, c="darkorchid", label="RFR")
ax[1].scatter(vv_const, yplus_DNS_test, marker="o", s=10, c="orange", label="Konstant c")
ax[1].set_xlabel("$\overline{v'v'}^+$", fontsize=14)
ax[1].set_xlim(-3,2)
ax[1].legend(loc="best", fontsize=12)

ax[2].plot(ww_DNS, yplus_DNS, 'b-', c="deepskyblue", label="DNS-data")
ax[2].scatter(ww_NN, yplus_DNS_test, marker="o", s=10, c="orangered", label="Neuralt nätverk")
ax[2].scatter(ww_XGB, yplus_DNS_test, marker="o", s=10, c="darkorchid", label="RFR")
ax[2].scatter(ww_const, yplus_DNS_test, marker="o", s=10, c="orange", label="Konstant c")
ax[2].set_xlabel("$\overline{w'w'}^+$", fontsize=14)
ax[2].set_xlim(-1,3)
ax[2].set_ylim(0,2100)
ax[2].legend(loc="best", fontsize=12)

plt.show()
