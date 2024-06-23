from Process_Square150 import *
import numpy as np
import matplotlib.pyplot as plt
import torch
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
neural_net = torch.load('Square1000depth2width11epochs1000batchsize5rate5e-05.pt')
preds = neural_net(X_test_tensor)

c_NN =np.tan(preds.detach().numpy())
c1_NN = c_NN[:,0]
c2_NN = c_NN[:,1]
c3_NN = c_NN[:,2]
c4_NN = c_NN[:,3]
c5_NN = c_NN[:,4]

c_constant = [-0.1, 0.1, 0.26, -10 * cmu ** 2, 0]

#c = [0.035,0.17,-0.14,0,0]

a11_NN = np.array([Astack_DNS_test[i].dot(c_NN[i])[0] for i in range(len(Astack_DNS_test))])
a12_NN = np.array([Astack_DNS_test[i].dot(c_NN[i])[1] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s12_DNS_test
a13_NN = np.array([Astack_DNS_test[i].dot(c_NN[i])[2] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s13_DNS_test
a22_NN = np.array([Astack_DNS_test[i].dot(c_NN[i])[3] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s22_DNS_test
a23_NN = np.array([Astack_DNS_test[i].dot(c_NN[i])[4] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s23_DNS_test

a11_constant = np.array([Astack_DNS_test[i].dot(c_constant)[0] for i in range(len(Astack_DNS_test))])
a12_constant = np.array([Astack_DNS_test[i].dot(c_constant)[1] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s12_DNS_test
a13_constant = np.array([Astack_DNS_test[i].dot(c_constant)[2] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s13_DNS_test
a22_constant = np.array([Astack_DNS_test[i].dot(c_constant)[3] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s22_DNS_test
a23_constant = np.array([Astack_DNS_test[i].dot(c_constant)[4] for i in range(len(Astack_DNS_test))])-2*cmu*tau_DNS_test*s23_DNS_test

uu_NN = (a11_NN+2/3)*k_DNS_test
uv_NN = a12_NN*k_DNS_test
uw_NN = a13_NN*k_DNS_test
vv_NN = (a22_NN+2/3)*k_DNS_test
vw_NN = a23_NN*k_DNS_test


uu_constant = (a11_constant+2/3)*k_DNS_test
uv_constant = a12_constant*k_DNS_test
uw_constant = a13_constant*k_DNS_test
vv_constant = (a22_constant+2/3)*k_DNS_test
vw_constant = a23_constant*k_DNS_test

variables = ['c1','c2','c3','c4','c5','uu', 'uv', 'uw', 'vv', 'vw']
models = ['NN', 'constant']
metrics = ['MAPE','RMSPE','MaAE']

# Outer loop for variables
for var in variables:
    # Inner loop for models
    for model in models:
        # Load data for DNS and model output
        try:
            dns_data = locals()[f'{var}_DNS_test']
            model_data = locals()[f'{var}_{model}']

            # Calculate metrics
            for metric in metrics:
                if metric == 'RMSPE':
                    value = np.sqrt(np.mean(np.square(((dns_data - model_data) / dns_data)), axis=0))
                elif metric == 'MAPE':
                    value = np.mean(np.abs((dns_data - model_data) / dns_data))
                elif metric == 'MaAE':
                    value = np.max(np.abs(dns_data - model_data))
                if metric != 'MaAE':
                    value = round(value, 2)

                # Print or use the results
                print(f'{var}_{metric}_{model}: {value}')
        except:
            print(str(var)+' average fel finns ej för konstant c')

plt.scatter(uu_DNS_test,yplus_DNS_test,label='DNS')
plt.scatter(uu_constant,yplus_DNS_test,s=5,label='constant')
plt.scatter(uu_NN,yplus_DNS_test,s=2,label='NN',c='r')

plt.xlabel("$\overline{uu}$")
plt.ylabel("$y^+$")
plt.legend()
plt.show()

plt.scatter(uv_DNS_test,yplus_DNS_test,label='DNS')
plt.scatter(uv_constant,yplus_DNS_test,s=5,label='constant c')
plt.scatter(uv_NN,yplus_DNS_test,s=2,label='NN',c='r')
plt.xlabel("$\overline{uv}$")
plt.ylabel("$y^+$")
plt.legend()
plt.show()

plt.scatter(uw_DNS_test,yplus_DNS_test,label='DNS')
plt.scatter(uw_constant,yplus_DNS_test,s=5,label='constant c')
plt.scatter(uw_NN,yplus_DNS_test,s=2,label='NN',c='r')
plt.xlabel("$\overline{uw}$")
plt.ylabel("$y^+$")
plt.legend()
plt.show()

plt.scatter(vv_DNS_test,yplus_DNS_test,label='DNS')
plt.scatter(vv_constant,yplus_DNS_test,s=5,label='constant c')
plt.scatter(vv_NN,yplus_DNS_test,s=2,label='NN',c='r')
plt.xlabel("$\overline{vv}$")
plt.ylabel("$y^+$")
plt.legend()
plt.show()

plt.scatter(vw_DNS_test,yplus_DNS_test,label='DNS')
plt.scatter(vw_constant,yplus_DNS_test,s=5,label='constant c')
plt.scatter(vw_NN,yplus_DNS_test,s=2,label='NN',c='r')
plt.xlabel("$\overline{vw}$")
plt.ylabel("$y^+$")
plt.legend()
plt.show()