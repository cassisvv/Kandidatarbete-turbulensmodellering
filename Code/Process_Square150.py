import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from pathlib import Path
from scipy.interpolate import griddata

data_path = Path('data')


# ------------------load DNS data---------------------------------------------
bound_data_path = data_path / 'SQUARE_DUCT_DATA' / 'RETAU500' / 'stress_1.dat'
DNS=np.genfromtxt(bound_data_path, dtype=None, skip_header=1)

yplus_DNS = DNS[:, 0]
zplus_DNS = DNS[:, 1]
u_DNS = DNS[:, 2]
v_DNS = DNS[:, 3]
w_DNS = DNS[:, 4]

utau_DNS = DNS[:, 29]
deltav_DNS = DNS[:, 30]

eps_DNS = DNS[:, 22]*utau_DNS**3/deltav_DNS
k_DNS   = DNS[:, 23]*utau_DNS**2
tau_DNS = k_DNS / eps_DNS

a11_DNS = DNS[:, 10]*utau_DNS**2/k_DNS
a12_DNS = DNS[:, 11]*utau_DNS**2/k_DNS
a13_DNS = DNS[:, 12]*utau_DNS**2/k_DNS
a22_DNS = DNS[:, 13]*utau_DNS**2/k_DNS
a23_DNS = DNS[:, 14]*utau_DNS**2/k_DNS
a33_DNS = DNS[:, 15]*utau_DNS**2/k_DNS

uu_DNS = (a11_DNS + 2 / 3) * k_DNS
uv_DNS = a12_DNS * k_DNS
uw_DNS = a13_DNS * k_DNS
vv_DNS = (a22_DNS + 2 / 3) * k_DNS
vw_DNS = a23_DNS * k_DNS
ww_DNS = (a33_DNS + 2 / 3) * k_DNS

y_grid = np.unique(yplus_DNS)
z_grid = np.unique(zplus_DNS)
Y, Z = np.meshgrid(y_grid, z_grid)

U = griddata((yplus_DNS, zplus_DNS), u_DNS, (Y, Z), method='linear')
V = griddata((yplus_DNS, zplus_DNS), v_DNS, (Y, Z), method='linear')
W = griddata((yplus_DNS, zplus_DNS), w_DNS, (Y, Z), method='linear')

dudy_DNS, dudz_DNS = np.gradient(U, y_grid, z_grid, axis=(1, 0))
dvdy_DNS, dvdz_DNS = np.gradient(V, y_grid, z_grid, axis=(1, 0))
dwdy_DNS, dwdz_DNS = np.gradient(W, y_grid, z_grid, axis=(1, 0))

dudy_DNS = dudy_DNS.flatten()
dudz_DNS = dudz_DNS.flatten()
dvdy_DNS = dvdy_DNS.flatten()
dvdz_DNS = dvdz_DNS.flatten()
dwdy_DNS = dwdy_DNS.flatten()
dwdz_DNS = dwdz_DNS.flatten()

s12_DNS = utau_DNS*dudy_DNS / 2
s13_DNS = utau_DNS*dudz_DNS / 2
s22_DNS = utau_DNS*dvdy_DNS
s23_DNS = utau_DNS*(dvdz_DNS + dwdy_DNS) / 2
s33_DNS = utau_DNS*dwdz_DNS

w12_DNS = utau_DNS*dudy_DNS / 2
w13_DNS = utau_DNS*dudz_DNS / 2
w23_DNS = utau_DNS*(dvdz_DNS - dwdy_DNS) / 2

cmu = 0.09

def smatrix(s12,s13,s22,s23,s33):
    return np.array([[0,s12,s13],[s12,s22,s23],[s13,s23,s33]])

def wmatrix(w12,w13,w23):
    return np.array([[0,w12,w13],[-w12,0,w23],[-w13,-w23,0]])

def Amatrix(s,w,t):
    A = np.zeros((5,5))
    delta = np.identity(3)
    index = [(0,0,0),(1,0,1),(2,0,2),(3,1,1),(4,1,2)]
    for q,i,j in index:
        A[q,0] = t**2*(sum(s[i, k] * s[k, j] for k in range(3)) - sum(1/3 * s[l, k] * s[l, k] * delta[i, j] for k in range(3) for l in range(3)))
        A[q,1] = t**2*sum(w[i,k]*s[k,j]-s[i,k]*w[k,j] for k in range(3))
        A[q,2] = t**2*(sum(w[i,k]*w[j,k] for k in range(3))-sum(1/3*w[l,k]*w[l,k]*delta[i,j] for k in range(3) for l in range(3)))
        A[q,3] = t**3*sum(s[i,k]*s[k,l]*w[l,j]-w[i,l]*s[l,k]*s[k,j] for k in range(3) for l in range(3))
        A[q,4] = t**3*(sum(w[i,l]*w[l,m]*s[m,j]+s[i,l]*w[l,m]*w[m,j] for l in range(3) for m in range(3)) -sum(2/3*w[m,n]*w[n,l]*s[l,m]*delta[i,j] for n in range(3) for l in range(3) for m in range(3)))
    return A

def aminusblist(a11,a12,a13,a22,a23,s12,s13,s22,s23,t):
    return [a11+2*cmu*t*0, a12+2*cmu*t*s12, a13+2*cmu*t*s13, a22+2*cmu*t*s22, a23+2*cmu*t*s23]

def makeb(s12,s13,s22,s23,t):
    return [-2*cmu*t*0, -2*cmu*t*s12, -2*cmu*t*s13, -2*cmu*t*s22, -2*cmu*t*s23]

Astack_DNS = np.stack([Amatrix(smatrix(s12_DNS[i], s13_DNS[i], s22_DNS[i], s23_DNS[i], s33_DNS[i]), wmatrix(w12_DNS[i], w13_DNS[i], w23_DNS[i]), tau_DNS[i]) for i in range(len(tau_DNS))])
aminusbstack_DNS = np.stack([aminusblist(a11_DNS[i], a12_DNS[i], a13_DNS[i], a22_DNS[i], a23_DNS[i], s12_DNS[i], s13_DNS[i], s22_DNS[i], s23_DNS[i], tau_DNS[i]) for i in range(len(tau_DNS))])
bstack_DNS = np.stack([makeb(s12_DNS[i], s13_DNS[i], s22_DNS[i], s23_DNS[i], tau_DNS[i]) for i in range(len(tau_DNS))])
c_DNS = np.linalg.solve(Astack_DNS, aminusbstack_DNS)


#----------------------data manipulation----------------------
#cut out center if needed:
#index_choose1=np.nonzero((yplus_DNS > -0.7) & (yplus_DNS < 0.7))[0]
#index_choose2=list(set(np.nonzero((yplus_DNS < -0.3))[0]).union(np.nonzero((yplus_DNS > 0.3))[0]).union(list(set(np.nonzero((zplus_DNS < -0.3))[0]).union(np.nonzero((zplus_DNS > 0.3))[0]))))
#index_choose3=np.nonzero((zplus_DNS > -0.7) & (zplus_DNS < 0.7))[0]
#index_choose4=list(set(np.nonzero((zplus_DNS < -0.3))[0]).union(np.nonzero((zplus_DNS > 0.3))[0]))
#index_choose = list(set(index_choose1).intersection(index_choose2).intersection(index_choose3))

#choose values with -0.75<y+<0.75, -0.75<z+<0.75
index_choose1=np.nonzero((yplus_DNS > -0.6) & (yplus_DNS < 0.6))[0]
index_choose2=np.nonzero((zplus_DNS > -0.6) & (zplus_DNS < 0.6))[0]
index_choose = list(set(index_choose1).intersection(index_choose2))

yplus_DNS = yplus_DNS[index_choose]
zplus_DNS = zplus_DNS[index_choose]
u_DNS = u_DNS[index_choose]
v_DNS = v_DNS[index_choose]
w_DNS = w_DNS[index_choose]

a11_DNS = a11_DNS[index_choose]
a12_DNS = a12_DNS[index_choose]
a13_DNS = a13_DNS[index_choose]
a22_DNS = a22_DNS[index_choose]
a23_DNS = a23_DNS[index_choose]
a33_DNS = a33_DNS[index_choose]

utau_DNS = utau_DNS[index_choose]
deltav_DNS = deltav_DNS[index_choose]

eps_DNS = eps_DNS[index_choose]
k_DNS   = k_DNS[index_choose]
tau_DNS = tau_DNS[index_choose]

uu_DNS = uu_DNS[index_choose]
uv_DNS = uv_DNS[index_choose]
uw_DNS = uw_DNS[index_choose]
vv_DNS = vv_DNS[index_choose]
vw_DNS = vw_DNS[index_choose]
ww_DNS = ww_DNS[index_choose]

dudy_DNS = dudy_DNS[index_choose]
dudz_DNS = dudz_DNS[index_choose]
dvdy_DNS = dvdy_DNS[index_choose]
dvdz_DNS = dvdz_DNS[index_choose]
dwdy_DNS = dwdy_DNS[index_choose]
dwdz_DNS = dwdz_DNS[index_choose]

s12_DNS = s12_DNS[index_choose]
s13_DNS = s13_DNS[index_choose]
s22_DNS = s22_DNS[index_choose]
s23_DNS = s23_DNS[index_choose]
s33_DNS = s33_DNS[index_choose]

w12_DNS = w12_DNS[index_choose]
w13_DNS = w13_DNS[index_choose]
w23_DNS = w23_DNS[index_choose]

Astack_DNS = Astack_DNS[index_choose]
aminusbstack_DNS = aminusbstack_DNS[index_choose]
c_DNS = c_DNS[index_choose]

# set a min on gradients
dudy_DNS = np.maximum(dudy_DNS, 4e-4)
dudz_DNS = np.maximum(dudz_DNS, 4e-4)
dvdy_DNS = np.maximum(dvdy_DNS, 4e-4)
dvdz_DNS = np.maximum(dvdz_DNS, 4e-4)
dwdy_DNS = np.maximum(dwdy_DNS, 4e-4)
dwdz_DNS = np.maximum(dwdz_DNS, 4e-4)


#----------------------training variables--------------------------

# transpose the target vector to make it a column vector
y = np.arctan(c_DNS)

dudy_squared_DNS = (dudy_DNS ** 2)
dudz_squared_DNS = (dudz_DNS ** 2)
dvdy_squared_DNS = (dvdy_DNS ** 2)
dvdz_squared_DNS = (dvdz_DNS ** 2)
dwdy_squared_DNS = (dwdy_DNS ** 2)
dwdz_squared_DNS = (dwdz_DNS ** 2)

# scale with k and eps
# dudy [1/T]
# dudy**2 [1/T**2]
T = tau_DNS #?????????? stämmer detta??????

dudy_squared_scaled = dudy_squared_DNS * T ** 2
dudz_squared_scaled = dudz_squared_DNS * T ** 2
dvdy_squared_scaled = dvdy_squared_DNS * T ** 2
dvdz_squared_scaled = dvdz_squared_DNS * T ** 2
dwdy_squared_scaled = dwdy_squared_DNS * T ** 2
dwdz_squared_scaled = dwdz_squared_DNS * T ** 2

dudy_DNS_inv = 1 / dudy_DNS / T
dudz_DNS_inv = 1 / dudz_DNS / T
dvdy_DNS_inv = 1 / dvdy_DNS / T
dvdz_DNS_inv = 1 / dvdz_DNS / T
dwdy_DNS_inv = 1 / dwdy_DNS / T
dwdz_DNS_inv = 1 / dwdz_DNS / T
# re-shape

dudy_squared_scaled = dudy_squared_scaled.reshape(-1,1)
dudz_squared_scaled = dudz_squared_scaled.reshape(-1,1)
dvdy_squared_scaled = dvdy_squared_scaled.reshape(-1,1)
dvdz_squared_scaled = dvdz_squared_scaled.reshape(-1,1)
dwdy_squared_scaled = dwdy_squared_scaled.reshape(-1,1)
dwdz_squared_scaled = dwdz_squared_scaled.reshape(-1,1)

dudy_inv_scaled = dudy_DNS_inv.reshape(-1,1)
dudz_inv_scaled = dudz_DNS_inv.reshape(-1,1)
dvdy_inv_scaled = dvdy_DNS_inv.reshape(-1,1)
dvdz_inv_scaled = dvdz_DNS_inv.reshape(-1,1)
dwdy_inv_scaled = dwdy_DNS_inv.reshape(-1,1)
dwdz_inv_scaled = dwdz_DNS_inv.reshape(-1,1)

y_input = yplus_DNS.reshape(-1, 1)
z_input = zplus_DNS.reshape(-1, 1)
# use MinMax scaler
#scaler_dudy2 = StandardScaler()
#scaler_tau = StandardScaler()
scaler_dudy2 = MinMaxScaler()
scaler_dudy = MinMaxScaler()

X=np.zeros((len(dudy_DNS), 25))
index=0
for i in range(5):
    for j in range(5):
        Ainput = Astack_DNS[:,i,j].reshape(-1,1)
        X[:,index] = scaler_dudy2.fit_transform(np.arctan(Ainput))[:, 0]
        index+=1

s12_input = s12_DNS/utau_DNS
s13_input = s13_DNS/utau_DNS
s22_input = s22_DNS/utau_DNS
s23_input = s23_DNS/utau_DNS
s33_input = s33_DNS/utau_DNS
s12_input = s12_input.reshape(-1,1)
s13_input = s13_input.reshape(-1,1)
s22_input = s22_input.reshape(-1,1)
s23_input = s23_input.reshape(-1,1)
s33_input = s33_input.reshape(-1,1)

#X[:,0] = scaler_dudy2.fit_transform(np.arctan(A43_input))[:,0]
#X[:,1] = scaler_dudy2.fit_transform(np.arctan(A32_input))[:,0]

#X[:,2] = scaler_dudy2.fit_transform(A30_input)[:,0]
#X[:,3] = scaler_dudy2.fit_transform(A42_input)[:,0]
#X[:,4] = scaler_dudy2.fit_transform(A41_input)[:,0]
#X[:,5] = scaler_dudy2.fit_transform(A40_input)[:,0]

#X[:,0] = scaler_dudy2.fit_transform(s12_input)[:,0]
#X[:,1] = scaler_dudy2.fit_transform(s13_input)[:,0]
#X[:,2] = scaler_dudy2.fit_transform(s22_input)[:,0]
#X[:,3] = scaler_dudy2.fit_transform(s23_input)[:,0]
#X[:,4] = scaler_dudy2.fit_transform(s33_input)[:,0]


#X[:,0] = scaler_dudy2.fit_transform(dudy_squared_scaled)[:,0]
#X[:,1] = scaler_dudy2.fit_transform(dudz_squared_scaled)[:,0]
#X[:,2] = scaler_dudy2.fit_transform(dvdy_squared_scaled)[:,0]
#X[:,3] = scaler_dudy2.fit_transform(dvdz_squared_scaled)[:,0]
#X[:,4] = scaler_dudy2.fit_transform(dwdy_squared_scaled)[:,0]
#X[:,5] = scaler_dudy2.fit_transform(dwdz_squared_scaled)[:,0]
#X[:,5] = scaler_dudy.fit_transform(y_input)[:,0]
#X[:,6] = scaler_dudy.fit_transform(z_input)[:,0]

#X[:,8] = scaler_dudy.fit_transform(dudy_inv_scaled)[:,0]
#X[:,9] = scaler_dudy.fit_transform(dudz_inv_scaled)[:,0]
#X[:,10] = scaler_dudy.fit_transform(dvdy_inv_scaled)[:,0]
#X[:,11] = scaler_dudy.fit_transform(dvdz_inv_scaled)[:,0]
#X[:,12] = scaler_dudy.fit_transform(dwdy_inv_scaled)[:,0]
#X[:,13] = scaler_dudy.fit_transform(dwdz_inv_scaled)[:,0]




# split the feature matrix and target vector into training and validation sets
# test_size=0.2 means we reserve 20% of the data for validation
# random_state=42 is a fixed seed for the random number generator, ensuring reproducibility

random_state = randrange(100)

indices = np.arange(len(X))
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indices, test_size=0.2, shuffle=True, random_state=42)

dudy_DNS_train = dudy_DNS[index_train]
dudz_DNS_train = dudz_DNS[index_train]
dvdy_DNS_train = dvdy_DNS[index_train]
dvdz_DNS_train = dvdz_DNS[index_train]
dwdy_DNS_train = dwdy_DNS[index_train]
dwdz_DNS_train = dwdz_DNS[index_train]

dudy_DNS_test = dudy_DNS[index_test]
dudz_DNS_test = dudz_DNS[index_test]
dvdy_DNS_test = dvdy_DNS[index_test]
dvdz_DNS_test = dvdz_DNS[index_test]
dwdy_DNS_test = dwdy_DNS[index_test]
dwdz_DNS_test = dwdz_DNS[index_test]
bstack_DNS_test = bstack_DNS[index_test,:]

c_DNS_train_dict = {'c1':c_DNS[index_train,0], 'c2':c_DNS[index_train,1], 'c3':c_DNS[index_train,2], 'c4':c_DNS[index_train,3], 'c5':c_DNS[index_train,4]}
c_DNS_test_dict =  {'c1':c_DNS[index_test,0], 'c2':c_DNS[index_test,1], 'c3':c_DNS[index_test,2], 'c4':c_DNS[index_test,3], 'c5':c_DNS[index_test,4]}
c_DNS_test_list = c_DNS[index_test, :]
c_DNS_train_list = c_DNS[index_train, :]
Astack_DNS_test = Astack_DNS[index_test]
Astack_DNS_train = Astack_DNS[index_train]
yplus_DNS_test = yplus_DNS[index_test]
yplus_DNS_train = yplus_DNS[index_train]
zplus_DNS_test = zplus_DNS[index_test]
zplus_DNS_train = zplus_DNS[index_train]
k_DNS_test = k_DNS[index_test]
uu_DNS_test = uu_DNS[index_test]
uv_DNS_test = uv_DNS[index_test]
uw_DNS_test = uw_DNS[index_test]
vv_DNS_test = vv_DNS[index_test]
vw_DNS_test = vw_DNS[index_test]
a11_DNS_test = a11_DNS[index_test]
a11_DNS_train = a11_DNS[index_train]

tau_DNS_test = tau_DNS[index_test]
s12_DNS_test = s12_DNS[index_test]
s13_DNS_test = s13_DNS[index_test]
s22_DNS_test = s22_DNS[index_test]
s23_DNS_test = s23_DNS[index_test]
"""
#dudy_DNS_inv_train = dudy_DNS_inv[index_train]
k_train = k_DNS[index_train]
uu_train = uu_DNS[index_train]
vv_train = vv_DNS[index_train]
ww_train = ww_DNS[index_train]
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
"""