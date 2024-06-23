from Process_Hill import y_test, tau_DNS_test, c1_DNS_test, c2_DNS_test, c3_DNS_test, uu_DNS_test, y_DNS_test, \
    X_test, y_test, y_DNS_test, x_DNS_test, w12_DNS_test, w21_DNS_test, s11_DNS_test, s12_DNS_test, s21_DNS_test, s22_DNS_test, \
    tau_DNS_test, uv_DNS_test, vv_DNS_test, ww_DNS_test, a11_DNS_test, a22_DNS_test,a12_DNS_test, a33_DNS_test, k_DNS_test
from random_forest import c_RFR
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
print(X_test.shape)
# make tensors for NN
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
neural_net = torch.load('Hilldepth2width10epochs10batchsize10rate0.0001.pt')
preds = neural_net(X_test_tensor)

# NN prediction
c_NN = preds.detach().numpy()
c1_NN = c_NN[:, 0]
c2_NN = c_NN[:, 1]
c3_NN = c_NN[:, 2]
uu_NN = c_NN[:, 3]
uv_NN = c_NN[:, 4]
vv_NN = c_NN[:, 5]
ww_NN = c_NN[:, 6]


# load XGB_model
XGB_model_c1 = pickle.load(open('XGB_2D_c1.pkl', 'rb'))
XGB_model_c2 = pickle.load(open('XGB_2D_c2.pkl', 'rb'))
XGB_model_c3 = pickle.load(open('XGB_2D_c3.pkl', 'rb'))
XGB_model_uu = pickle.load(open('XGB_2D_uu.pkl', 'rb'))
XGB_model_vv = pickle.load(open('XGB_2D_vv.pkl', 'rb'))
XGB_model_ww = pickle.load(open('XGB_2D_ww.pkl', 'rb'))
XGB_model_uv = pickle.load(open('XGB_2D_uv.pkl', 'rb'))

# XGB prediction
c1_XGB = XGB_model_c1.predict(X_test)
c2_XGB = XGB_model_c2.predict(X_test)
c3_XGB = XGB_model_c3.predict(X_test)
uu_XGB = XGB_model_uu.predict(X_test)
vv_XGB = XGB_model_vv.predict(X_test)
ww_XGB = XGB_model_ww.predict(X_test)
uv_XGB = XGB_model_uv.predict(X_test)


def calculate_aij(c1, c2, c3):

    c_mu = 0.09
    c4 = 0
    c5 = 0
    c6 = 0
    c7 = 0
    tau = tau_DNS_test

    s11 = s11_DNS_test
    s12 = s12_DNS_test
    s21 = s21_DNS_test
    s22 = s22_DNS_test
    w12 = w12_DNS_test
    w21 = w21_DNS_test

    # Calculate terms for a11
    a11 = -2 * c_mu * s11 * tau
    a11 += (1/3) * c1 * tau**2 * (2*s11*s11 + s12*s12 - s22*s22)
    a11 += 2 * c2 * tau**2 * s12 * w12
    a11 += (1/3) * c3 * tau**2 * w12**2
    a11 -= 2 * c4 * tau**2 * s12 * w12 * (s11 + s22)
    a11 += c5 * tau**3 * w12**2 * (-4/3 * s11 + 2/3 * s22)
    a11 += c6 * tau**3 * s11 * (s11*s11 + 2*s12*s12 + s22*s22)
    a11 += 2 * c7 * tau**3 * s11 * w12**2
    uu = (a11 + 0.6666) * k_DNS_test

    # Calculate terms for a12
    a12 = -2 * c_mu * s12 * tau
    a12 += c1 * tau**2 * s12 * (s11 + s22)
    a12 += c2 * tau**2 * w12 * (-s11 + s22)
    a12 += c4 * tau**2 * w12 * (s11**2 - s22**2)
    a12 -= 2 * c5 * tau**3 * s12 * w12**2
    a12 += c6 * tau**3 * s12 * (s11*s11 + 2*s12*s12 + s22*s22)
    a12 += 2 * c7 * tau**3 * s12 * w12**2
    uv = a12 * k_DNS_test

    # Calculate terms for a22
    a22 = -2 * c_mu * s22 * tau
    a22 += (1/3) * c1 * tau**2 * (-s11**2 + s12**2 + 2*s22**2)
    a22 -= 2 * c2 * tau**2 * s12 * w12
    a22 += (1/3) * c3 * tau**2 * w12**2
    a22 += 2 * c4 * tau**2 * s12 * w12 * (s11 + s22)
    a22 += (2/3) * c5 * tau**3 * w12**2 * (s11 - 2*s22)
    a22 += c6 * tau**3 * s22 * (s11*s11 + 2*s12*s12 + s22*s22)
    a22 += 2 * c7 * s22 * tau**3 * w12**2
    vv = (a22 + 0.6666) * k_DNS_test

    # Calculate terms for a33
    a33 = -(1/3) * c1 * tau**2 * (s11**2 + 2*s12**2 + s22**2)
    a33 -= (2/3) * c3 * tau**2 * w12**2
    a33 += (2/3) * c5 * tau**3 * w12**2 * (s11 + s22)
    ww = (a33 + 0.6666) * k_DNS_test

    return a11, a12, a22, a33, uu, uv, vv, ww

#a11_NN, a12_NN, a22_NN, a33_NN, uu_NN, uv_NN, vv_NN, ww_NN  = calculate_aij(c1_NN, c2_NN, c3_NN)
#a11_DNS, a12_DNS, a22_DNS, a33_DNS, uu_DNS, uv_DNS, vv_DNS, ww_DNS = calculate_aij(c1_DNS_test, c2_DNS_test, c3_DNS_test)
#a11_RFR, a12_RFR, a22_RFR, a33_RFR = calculate_aij(c1_RFR, c2_RFR, c3_RFR)
#a11_XGB, a12_XGB, a22_XGB, a33_XGB, uu_XGB, uv_XGB, vv_XGB, ww_XGB = calculate_aij(c1_XGB, c2_XGB, c3_XGB)

# calculate and print all errors
variables = ['uu', 'uv', 'vv', 'ww', 'c1', 'c2', 'c3']
models = [ 'NN', 'XGB']
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




# Calculate the absolute difference simulated and true turbulence
uu_diff_XGB = (np.abs(uu_XGB - uu_DNS_test))
uv_diff_XGB = (np.abs(uv_XGB - uv_DNS_test))
vv_diff_XGB = (np.abs(vv_XGB - vv_DNS_test))
ww_diff_XGB = (np.abs(ww_XGB - ww_DNS_test))

uu_diff_NN = (np.abs(uu_NN - uu_DNS_test))
uv_diff_NN = (np.abs(uv_NN - uv_DNS_test))
vv_diff_NN = (np.abs(vv_NN - vv_DNS_test))
ww_diff_NN = (np.abs(ww_NN - ww_DNS_test))


print(uu_DNS_test, uu_NN)
# plot turbulence XGB vs DNS

# Create subplots
fig, ax = plt.subplots(2, 2, figsize=(9, 6), gridspec_kw={'hspace': 0.4, 'wspace': 0.2})

# Plot data on each subplot and add individual colorbars
scatter1 = ax[0, 0].scatter(x_DNS_test, y_DNS_test, s=5, c=uu_diff_XGB, cmap='jet')
ax[0, 0].set_title("Fel $\overline{u'u'}^+$")
cbar1 = fig.colorbar(scatter1, ax=ax[0, 0])

scatter2 = ax[0, 1].scatter(x_DNS_test, y_DNS_test, s=5, c=uv_diff_XGB, cmap='jet')
ax[0, 1].set_title("Fel $\overline{u'v'}^+$")
cbar2 = fig.colorbar(scatter2, ax=ax[0, 1])

scatter3 = ax[1, 0].scatter(x_DNS_test, y_DNS_test, s=5, c=vv_diff_XGB, cmap='jet')
ax[1, 0].set_title("Fel $\overline{v'v'}^+$")
cbar3 = fig.colorbar(scatter3, ax=ax[1, 0])

scatter4 = ax[1, 1].scatter(x_DNS_test, y_DNS_test, s=5, c=ww_diff_XGB, cmap='jet')
ax[1, 1].set_title("Fel $\overline{w'w'}^+$")
cbar4 = fig.colorbar(scatter4, ax=ax[1, 1])

# Set common labels
fig.text(0.5, 0.04, 'x [m]', ha='center', va='center')
fig.text(0.06, 0.5, 'y [m]', ha='center', va='center', rotation='vertical')

# Set the main title for the entire figure
fig.suptitle("Absoluta felet för prediktioner av stresstensorer med RFR", fontsize=18)

plt.show()


import matplotlib.pyplot as plt

# Create subplots
fig, ax = plt.subplots(2, 2, figsize=(9, 6), gridspec_kw={'hspace': 0.4, 'wspace': 0.2})

# Plot data on each subplot and add individual colorbars
scatter1 = ax[0, 0].scatter(x_DNS_test, y_DNS_test, s=5, c=uu_diff_NN, cmap='jet')
ax[0, 0].set_title("Fel $\overline{u'u'}^+$")
cbar1 = fig.colorbar(scatter1, ax=ax[0, 0])

scatter2 = ax[0, 1].scatter(x_DNS_test, y_DNS_test, s=5, c=uv_diff_NN, cmap='jet')
ax[0, 1].set_title("Fel $\overline{u'v'}^+$")
cbar2 = fig.colorbar(scatter2, ax=ax[0, 1])

scatter3 = ax[1, 0].scatter(x_DNS_test, y_DNS_test, s=5, c=vv_diff_NN, cmap='jet')
ax[1, 0].set_title("Fel $\overline{v'v'}^+$")
cbar3 = fig.colorbar(scatter3, ax=ax[1, 0])

scatter4 = ax[1, 1].scatter(x_DNS_test, y_DNS_test, s=5, c=ww_diff_NN, cmap='jet')
ax[1, 1].set_title("Fel $\overline{w'w'}^+$")
cbar4 = fig.colorbar(scatter4, ax=ax[1, 1])

# Set common labels
fig.text(0.5, 0.04, 'x [m]', ha='center', va='center')
fig.text(0.06, 0.5, 'y [m]', ha='center', va='center', rotation='vertical')

# Set the main title for the entire figure
fig.suptitle("Absoluta felet för prediktioner av stresstensorer med neuralt nätverk", fontsize=18)

plt.show()




'''
abs_diff = abs(vv_DNS_test-vv_XGB)
plt.scatter(x_DNS_test, y_DNS_test, s=20, c=abs_diff)
plt.xlabel('x')
plt.ylabel('y')
plt.title('')
plt.colorbar()
plt.show()
# Filter out outliers where abs(c1_NN - c1_DNS_test) > 2.5
filtered_indices = abs_diff <= 10e200
x_filtered = x_DNS_test[filtered_indices]
y_filtered = y_DNS_test[filtered_indices]
c_filtered = abs_diff[filtered_indices]

plt.hexbin(x_filtered, y_filtered, gridsize=50, C=c_filtered)
plt.colorbar()
plt.show()




# Create the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(c1_NN, extent=(np.min(x_DNS_test), np.max(x_DNS_test), np.min(y_DNS_test), np.max(y_DNS_test)), origin='lower', cmap='hot')
plt.colorbar(label='c')
plt.xlabel('xplus')
plt.ylabel('yplus')
plt.title('Heatmap of c vs xplus and yplus')
plt.grid(False)
plt.show()

# plot coefficients NN and RFR vs DNS
fig,axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
axs[1].plot(c[1,:], yplus_DNS,'b-', c="deepskyblue", label="DNS-data")
axs[1].scatter(c2_NN, y_DNS_test, marker="o", s=5, c="orangered", label="Neuralt nätverk")
axs[1].scatter(c2_RFR, y_DNS_test, marker="o", s=5, c="yellowgreen", label="RFR")
axs[1].legend(loc="best", fontsize=12)
axs[1].set_xlabel("$c_2$", fontsize=14)
axs[0].set_ylabel("$y^+$", fontsize=14)

axs[0].plot(c[0,:], yplus_DNS,'b-', c="deepskyblue", label="DNS-data")
axs[0].scatter(c0_NN, y_DNS_test, marker="o", s=5, c="orangered", label="Neuralt nätverk")
axs[0].scatter(c0_RFR, y_DNS_test, marker="o", s=5, c="yellowgreen", label="RFR")
axs[0].legend(loc="best", fontsize=12)
axs[0].set_xlabel("$c_0$", fontsize=14)
fig.suptitle("Prediktioner för koefficienterna $c_i$", fontsize=18)
plt.show()


# aij plot NN and RFR

yplus_DNS_sort, a_11_DNS_sort = zip(*sorted(zip(y_DNS_test, a_11_DNS_test)))

fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
ax[0].plot(a_11_DNS_sort, yplus_DNS_sort, c="deepskyblue", label="DNS-data")
ax[0].scatter(a_11_NN, y_DNS_test, marker="o", s=5, c="orangered", label="Neuralt nätverk")
ax[0].scatter(a_11_RFR, y_DNS_test, marker="o", s=5, c="yellowgreen", label="RFR")
ax[0].legend(loc="best", fontsize=12)
ax[0].set_xlabel("$a_{11}$", fontsize=14)
ax[0].set_ylabel("$y^+$", fontsize=14)
fig.suptitle("Prediktioner för $a_{ij}$", fontsize=18)

yplus_DNS_sort, a_22_DNS_sort = zip(*sorted(zip(y_DNS_test, a_22_DNS_test)))

ax[1].plot(a_22_DNS_sort, yplus_DNS_sort, c="deepskyblue", label="DNS-data")
ax[1].scatter(a_22_NN, y_DNS_test, marker="o", s=5, c="orangered", label="Neuralt nätverk")
ax[1].scatter(a_22_RFR, y_DNS_test, marker="o", s=5, c="yellowgreen", label="RFR")
ax[1].set_xlabel("$a_{22}$", fontsize=14)
ax[1].legend(loc="best", fontsize=12)

plt.show()


# plot turbulence NN and RFR vs DNS
fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
ax[0].plot(uu_DNS, yplus_DNS, 'b-', c="deepskyblue", label="DNS-data")
ax[0].scatter(uu_NN, y_DNS_test, marker="o", s=5, c="orangered", label="Neuralt nätverk")
ax[0].scatter(uu_RFR, y_DNS_test, marker="o", s=5, c="yellowgreen", label="RFR")
ax[0].legend(loc="best", fontsize=12)
ax[0].set_xlabel("$\overline{u'u'}^+$", fontsize=14)
ax[0].set_ylabel("$y^+$", fontsize=14)
fig.suptitle("Prediktioner för alla stresstensorer", fontsize=18)

ax[1].plot(vv_DNS, yplus_DNS, 'b-', c="deepskyblue", label="DNS-data")
ax[1].scatter(vv_NN, y_DNS_test, marker="o", s=5, c="orangered", label="Neuralt nätverk")
ax[1].scatter(vv_RFR, y_DNS_test, marker="o", s=5, c="yellowgreen", label="RFR")
ax[1].set_xlabel("$\overline{v'v'}^+$", fontsize=14)
ax[1].legend(loc="best", fontsize=12)

ax[2].plot(ww_DNS, yplus_DNS, 'b-', c="deepskyblue", label="DNS-data")
ax[2].scatter(ww_NN, y_DNS_test, marker="o", s=5, c="orangered", label="Neuralt nätverk")
ax[2].scatter(ww_RFR, y_DNS_test, marker="o", s=5, c="yellowgreen", label="RFR")
ax[2].set_xlabel("$\overline{w'w'}^+$", fontsize=14)
ax[2].legend(loc="best", fontsize=12)

plt.show()
'''