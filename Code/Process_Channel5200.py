import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from pathlib import Path
from matplotlib import pyplot as plt

data_path = Path('data')
channel_data_path = data_path / 'Channel flow'


# ------------------load DNS data---------------------------------------------
vel_DNS=np.genfromtxt(channel_data_path / "LM_Channel_5200_vel_fluc_prof.dat", dtype=None,comments="%")
mean_DNS = np.genfromtxt(channel_data_path / "LM_Channel_5200_mean_prof.dat",comments="%")
RSTE_DNS = np.genfromtxt(channel_data_path / "LM_Channel_5200_RSTE_k_prof.dat",comments="%")

y_DNS=vel_DNS[:,0]
yplus_DNS=vel_DNS[:,1]
uu_DNS=vel_DNS[:,2]
vv_DNS=vel_DNS[:,3]
ww_DNS=vel_DNS[:,4]
uv_DNS=vel_DNS[:,6]

u_DNS=mean_DNS[:,2]

dudy_DNS  = np.gradient(u_DNS,yplus_DNS)

k_DNS  = 0.5*(uu_DNS+vv_DNS+ww_DNS)



eps_DNS = RSTE_DNS[:,7]
yplus_DNS_uu = yplus_DNS

# fix wall
eps_DNS[0]=eps_DNS[1]


#----------------------data manipulation----------------------

#choose values with 9 < y+ < 2200
index_choose=np.nonzero((yplus_DNS > 9 )  & (yplus_DNS< 2200 ))

# set a min on dudy
dudy_DNS = np.maximum(dudy_DNS,4e-4)


uv_DNS    =  uv_DNS[index_choose]
uu_DNS    =  uu_DNS[index_choose]
vv_DNS    =  vv_DNS[index_choose]
ww_DNS    =  ww_DNS[index_choose]
k_DNS     =  k_DNS[index_choose]
eps_DNS   =  eps_DNS[index_choose]
dudy_DNS  =  dudy_DNS[index_choose]
yplus_DNS =  yplus_DNS[index_choose]
y_DNS     =  y_DNS[index_choose]
u_DNS     =  u_DNS[index_choose]

viscous_t = k_DNS**2/eps_DNS
tau       = viscous_t/abs(uv_DNS)


dudy_DNS_org = np.copy(dudy_DNS)

tau_DNS = k_DNS/eps_DNS


a11_DNS=uu_DNS/k_DNS-0.66666
a22_DNS=vv_DNS/k_DNS-0.66666
a33_DNS=ww_DNS/k_DNS-0.66666

c_2_DNS=(2*a11_DNS+a33_DNS)/tau_DNS**2/dudy_DNS**2
c_0_DNS=-6*a33_DNS/tau_DNS**2/dudy_DNS**2

c = np.array([c_0_DNS,c_2_DNS])


#----------------------training variables--------------------------

# transpose the target vector to make it a column vector
y = c.transpose()

dudy_squared_DNS = (dudy_DNS**2)
# scale with k and eps
# dudy [1/T]
# dudy**2 [1/T**2]
T = tau_DNS
dudy_squared_DNS_scaled = dudy_squared_DNS*T**2
dudy_DNS_inv = 1/dudy_DNS/T
# re-shape
dudy_squared_DNS_scaled = dudy_squared_DNS_scaled.reshape(-1,1)
dudy_DNS_inv_scaled = dudy_DNS_inv.reshape(-1,1)
# use MinMax scaler
#scaler_dudy2 = StandardScaler()
#scaler_tau = StandardScaler()
scaler_dudy2 = MinMaxScaler()
scaler_dudy = MinMaxScaler()
X=np.zeros((len(dudy_DNS),2))
X[:,0] = scaler_dudy2.fit_transform(dudy_squared_DNS_scaled)[:,0]
X[:,1] = scaler_dudy.fit_transform(dudy_DNS_inv_scaled)[:,0]


# split the feature matrix and target vector into training and validation sets
# test_size=0.2 means we reserve 20% of the data for validation
# random_state=42 is a fixed seed for the random number generator, ensuring reproducibility

random_state = randrange(100)

indices = np.arange(len(X))
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indices,test_size=0.2,shuffle=True,random_state=42)


dudy_DNS_train = dudy_DNS[index_train]
dudy_DNS_inv_train = dudy_DNS_inv[index_train]
k_DNS_train = k_DNS[index_train]
uu_DNS_train = uu_DNS[index_train]
vv_DNS_train = vv_DNS[index_train]
ww_DNS_train = ww_DNS[index_train]
yplus_DNS_train = yplus_DNS[index_train]
c0_DNS_train = c_0_DNS[index_train]
c2_DNS_train = c_2_DNS[index_train]
tau_DNS_train = tau_DNS[index_train]

dudy_DNS_test = dudy_DNS[index_test]
dudy_DNS_inv_test = dudy_DNS_inv[index_test]
k_DNS_test = k_DNS[index_test]
uu_DNS_test = uu_DNS[index_test]
vv_DNS_test = vv_DNS[index_test]
ww_DNS_test = ww_DNS[index_test]
yplus_DNS_test = yplus_DNS[index_test]
c0_DNS_test = c_0_DNS[index_test]
c2_DNS_test = c_2_DNS[index_test]
tau_DNS_test = tau_DNS[index_test]

c_DNS_train_dict = {'c0':c0_DNS_train,'c2':c2_DNS_train}
c_DNS_test_dict =  {'c0':c0_DNS_test,'c2':c2_DNS_test}

