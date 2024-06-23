import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from pathlib import Path
from gradients import compute_face_phi,dphidx,dphidy,init

viscos=1/550

data_path = Path('data')
data_path = Path('data')
wave_data_path = data_path / 'Wave'
#region
datax= np.loadtxt(wave_data_path / "x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt(wave_data_path / "y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

x=xp2d[:,0]
y=yp2d[0,:]

ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
dist_s=yp2d-ywall_s[:,None]

#z grid
zmax, nk=np.loadtxt(wave_data_path / 'z.dat')
nk=int(nk)
zp = np.linspace(0, zmax, nk)

itstep,nk,dz=np.load(wave_data_path / 'itstep.npy')
p2d=np.load(wave_data_path / 'p_averaged.npy')/itstep
u2d=np.load(wave_data_path / 'u_averaged.npy')/itstep
v2d=np.load(wave_data_path / 'v_averaged.npy')/itstep
uu2d=np.load(wave_data_path / 'uu_stress.npy')/itstep
vv2d=np.load(wave_data_path / 'vv_stress.npy')/itstep
ww2d=np.load(wave_data_path / 'ww_stress.npy')/itstep
uv2d=np.load(wave_data_path / 'uv_stress.npy')/itstep

uu2d=uu2d-u2d**2
vv2d=vv2d-v2d**2
uv2d=uv2d-u2d*v2d

ubulk = np.trapz(u2d[0,:],yp2d[0,:])/max(y2d[0,:])
ustar2=viscos*u2d[:,0]/dist_s[:,0]
yplus2d=np.ones((ni,nj))
for i in range(0,ni):
   yplus2d[i,:]=(abs(ustar2[i]))**0.5*yp2d[i,:]/viscos
cf=(abs(ustar2))**0.5*np.sign(ustar2)/ubulk**2/0.5
ustar=(abs(ustar2))**0.5

kres_2d=0.5*(uu2d+vv2d+ww2d)

itstep_diss=np.load(wave_data_path / 'itstep_dissipation.npy')
eps2d= np.load(wave_data_path / 'diss_visc_mean.npy')/itstep_diss
dudx_mean = np.load(wave_data_path / 'dudx.npy')/itstep_diss
dudy_mean = np.load(wave_data_path / 'dudy.npy')/itstep_diss
dvdx_mean = np.load(wave_data_path / 'dvdx.npy')/itstep_diss
dvdy_mean = np.load(wave_data_path / 'dvdy.npy')/itstep_diss

eps2d = eps2d - viscos*(dudx_mean**2 + dudy_mean**2 + dvdx_mean**2 + dvdy_mean**2)

# compute re_delta1 for boundary layer flow
dx=x[3]-x[2]
re_disp_bl=np.zeros(ni)
delta_disp=np.zeros(ni)
for i in range (0,ni-1):
   d_disp=0
   for j in range (1,nj-1):
      up=u2d[i,j]/u2d[i,-1]
      dy=y2d[i,j]-y2d[i,j-1]
      d_disp=d_disp+(1.-min(up,1.))*dy

   delta_disp[i]=d_disp
   re_disp_bl[i]=d_disp*u2d[i,-1]/viscos

re_disp_bl[-1]=re_disp_bl[-1-1]
delta_disp[-1]=delta_disp[-1-1]

# compute geometric quantities
areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,as_bound = init(x2d,y2d,xp2d,yp2d)

# compute face value of U and V
zero_bc=np.zeros(ni)
u2d_face_w,u2d_face_s=compute_face_phi(u2d,fx,fy,ni,nj,zero_bc)
v2d_face_w,v2d_face_s=compute_face_phi(v2d,fx,fy,ni,nj,zero_bc)
p2d_face_w,p2d_face_s=compute_face_phi(p2d,fx,fy,ni,nj,p2d[:,0])

# x derivatives
dudx=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)
dpdx=dphidx(p2d_face_w,p2d_face_s,areawx,areasx,vol)

# y derivatives
dudy=dphidy(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy=dphidy(v2d_face_w,v2d_face_s,areawy,areasy,vol)
dpdy=dphidy(p2d_face_w,p2d_face_s,areawy,areasy,vol)
#endregion

y_DNS = yp2d.flatten()
x_DNS = xp2d.flatten()

uu_DNS = uu2d.flatten()
vv_DNS = vv2d.flatten()
ww_DNS = ww2d.flatten()
uv_DNS = uv2d.flatten()

dudx_DNS = dudx.flatten()
dudy_DNS = dudy.flatten()
dvdx_DNS = dvdx.flatten()
dvdy_DNS = dvdy.flatten()

eps_DNS = eps2d.flatten()
k_DNS = kres_2d.flatten()
tau_DNS = k_DNS/eps_DNS

s11_DNS = dudx_DNS
s12_DNS = 0.5 * (dudy_DNS + dvdx_DNS)
s21_DNS = -0.5 * (dudy_DNS + dvdx_DNS)
s22_DNS = dvdy_DNS
s33_DNS = 0 * dvdy_DNS

w12_DNS = 0.5 * (dudy_DNS - dvdx_DNS)
w21_DNS = -w12_DNS

cmu = 0.09

a11_DNS = uu_DNS / k_DNS - 2 / 3
a12_DNS = uv_DNS / k_DNS
a22_DNS = vv_DNS / k_DNS - 2 / 3
a33_DNS = ww_DNS / k_DNS - 2 / 3

c1_DNS = (2 * a11_DNS * s11_DNS - 2 * a11_DNS * s22_DNS + 4 * a12_DNS * s12_DNS + a33_DNS * s11_DNS - a33_DNS * s22_DNS + 4 * cmu * s11_DNS ** 2 * tau_DNS - 4 * cmu * s11_DNS * s22_DNS * tau_DNS + 8 * cmu * s12_DNS ** 2 * tau_DNS) / (s11_DNS ** 3 * tau_DNS ** 2 - s11_DNS ** 2 * s22_DNS * tau_DNS ** 2 + 4 * s11_DNS * s12_DNS ** 2 * tau_DNS ** 2 - s11_DNS * s22_DNS ** 2 * tau_DNS ** 2 + 4 * s12_DNS ** 2 * s22_DNS * tau_DNS ** 2 + s22_DNS ** 3 * tau_DNS ** 2)

c2_DNS = (2 * a11_DNS * s12_DNS - a12_DNS * s11_DNS + a12_DNS * s22_DNS + a33_DNS * s12_DNS + 2 * cmu * s11_DNS * s12_DNS * tau_DNS + 2 * cmu * s12_DNS * s22_DNS * tau_DNS) / (s11_DNS ** 2 * tau_DNS ** 2 * w12_DNS - 2 * s11_DNS * s22_DNS * tau_DNS ** 2 * w12_DNS + 4 * s12_DNS ** 2 * tau_DNS ** 2 * w12_DNS + s22_DNS ** 2 * tau_DNS ** 2 * w12_DNS)

c3_DNS = (-a11_DNS * s11_DNS ** 3 + a11_DNS * s11_DNS ** 2 * s22_DNS - 2 * a11_DNS * s11_DNS * s12_DNS ** 2 - a11_DNS * s11_DNS * s22_DNS ** 2 + 2 * a11_DNS * s12_DNS ** 2 * s22_DNS + a11_DNS * s22_DNS ** 3 - 2 * a12_DNS * s11_DNS ** 2 * s12_DNS - 4 * a12_DNS * s12_DNS ** 3 - 2 * a12_DNS * s12_DNS * s22_DNS ** 2 - 2 * a33_DNS * s11_DNS ** 3 + 2 * a33_DNS * s11_DNS ** 2 * s22_DNS - 7 * a33_DNS * s11_DNS * s12_DNS ** 2 + a33_DNS * s11_DNS * s22_DNS ** 2 - 5 * a33_DNS * s12_DNS ** 2 * s22_DNS -
          a33_DNS * s22_DNS ** 3 - 2 * cmu * s11_DNS ** 4 * tau_DNS + 2 * cmu * s11_DNS ** 3 * s22_DNS * tau_DNS - 8 * cmu * s11_DNS ** 2 * s12_DNS ** 2 * tau_DNS - 2 * cmu * s11_DNS ** 2 * s22_DNS ** 2 * tau_DNS + 4 * cmu * s11_DNS * s12_DNS ** 2 * s22_DNS * tau_DNS + 2 * cmu * s11_DNS * s22_DNS ** 3 * tau_DNS - 8 * cmu * s12_DNS ** 4 * tau_DNS - 4 * cmu * s12_DNS ** 2 * s22_DNS ** 2 * tau_DNS) / (s11_DNS ** 3 * tau_DNS ** 2 * w12_DNS ** 2 - s11_DNS ** 2 * s22_DNS * tau_DNS ** 2 * w12_DNS ** 2 + 4 * s11_DNS * s12_DNS ** 2 * tau_DNS ** 2 * w12_DNS ** 2 - s11_DNS * s22_DNS ** 2 * tau_DNS ** 2 * w12_DNS ** 2 + 4 * s12_DNS ** 2 * s22_DNS * tau_DNS ** 2 * w12_DNS ** 2 + s22_DNS ** 3 * tau_DNS ** 2 * w12_DNS ** 2)
"""
a11boi = -2 * cmu * s11_DNS * tau_DNS + \
         1 / 3 * c1_DNS * tau_DNS ** 2 * (2 * s11_DNS * s11_DNS + s12_DNS * s12_DNS - s22_DNS * s22_DNS) + \
         2 * c2_DNS * tau_DNS ** 2 * s12_DNS * w12_DNS + \
         1 / 3 * c3_DNS * tau_DNS ** 2 * w12_DNS * w12_DNS

a12boi = -2 * cmu * s12_DNS * tau_DNS + c1_DNS * tau_DNS ** 2 * s12_DNS * (s11_DNS + s22_DNS) + c2_DNS * tau_DNS ** 2 * w12_DNS * (-s11_DNS + s22_DNS)

a22boi = -2 * cmu * s22_DNS * tau_DNS + 1 / 3 * c1_DNS * tau_DNS ** 2 * (-s11_DNS * s11_DNS + s12_DNS * s12_DNS + 2 * s22_DNS * s22_DNS) - 2 * c2_DNS * tau_DNS ** 2 * s12_DNS * w12_DNS + 1 / 3 * c3_DNS * tau_DNS ** 2 * w12_DNS * w12_DNS

a33boi = -1 / 3 * c1_DNS * tau_DNS ** 2 * (s11_DNS * s11_DNS + 2 * s12_DNS * s12_DNS + s22_DNS * s22_DNS) - 2 / 3 * c3_DNS * tau_DNS ** 2 * w12_DNS * w12_DNS
"""
#----------------------data manipulation----------------------
'''
index_choose = np.nonzero((y_DNS > 0.2) & (y_DNS < 0.9))[0]

y_DNS = y_DNS[index_choose]
x_DNS = x_DNS[index_choose]

uu_DNS = uu_DNS[index_choose]
vv_DNS = vv_DNS[index_choose]
ww_DNS = ww_DNS[index_choose]
uv_DNS = uv_DNS[index_choose]

dudx_DNS = dudx_DNS[index_choose]
dudy_DNS = dudy_DNS[index_choose]
dvdx_DNS = dvdx_DNS[index_choose]
dvdy_DNS = dvdy_DNS[index_choose]

eps_DNS = eps_DNS[index_choose]
k_DNS = k_DNS[index_choose]
tau_DNS = tau_DNS[index_choose]

s11_DNS = s11_DNS[index_choose]
s12_DNS = s12_DNS[index_choose]
s21_DNS = s21_DNS[index_choose]
s22_DNS = s22_DNS[index_choose]
s33_DNS = s33_DNS[index_choose]

w12_DNS = w12_DNS[index_choose]
w21_DNS = w21_DNS[index_choose]

a11_DNS = a11_DNS[index_choose]
a12_DNS = a12_DNS[index_choose]
a22_DNS = a22_DNS[index_choose]
a33_DNS = a33_DNS[index_choose]

c1_DNS = c1_DNS[index_choose]
c2_DNS = c2_DNS[index_choose]
c3_DNS = c3_DNS[index_choose]
'''
c_DNS = np.array([c1_DNS,c2_DNS,c3_DNS, uu_DNS, uv_DNS, vv_DNS, ww_DNS])

#----------------------training variables--------------------------
y = np.arctan(c_DNS.transpose())

#A11_input = 1 / 3 * tau_DNS ** 2 * (2 * s11_DNS * s11_DNS + s12_DNS * s12_DNS - s22_DNS * s22_DNS)
A11_input = 1 / 3 * tau_DNS ** 2 * (2 * s11_DNS * s11_DNS + s12_DNS * s12_DNS - s22_DNS * s22_DNS)
A12_input = 2 * tau_DNS ** 2 * s12_DNS * w12_DNS
A13_input = 1 / 3 * tau_DNS ** 2 * w12_DNS * w12_DNS

A21_input = tau_DNS ** 2 * s12_DNS * (s11_DNS + s22_DNS)
A22_input = tau_DNS ** 2 * w12_DNS * (-s11_DNS + s22_DNS)

A31_input = -1 / 3 * tau_DNS ** 2 * (s11_DNS * s11_DNS + 2 * s12_DNS * s12_DNS + s22_DNS * s22_DNS)
A33_input = 2 / 3 * tau_DNS ** 2 * w12_DNS * w12_DNS

A11_input = A11_input.reshape(-1,1)
A12_input = A12_input.reshape(-1,1)
A13_input = A13_input.reshape(-1,1)
A21_input = A21_input.reshape(-1,1)
A22_input = A22_input.reshape(-1,1)
A31_input = A31_input.reshape(-1,1)
A33_input = A33_input.reshape(-1,1)

scaler = MinMaxScaler()
X=np.zeros((len(dudy_DNS), 7))
X[:,0] = scaler.fit_transform(np.arctan(A11_input))[:, 0]
X[:,1] = scaler.fit_transform(np.arctan(A12_input))[:, 0]
X[:,2] = scaler.fit_transform(np.arctan(A13_input))[:, 0]
X[:,3] = scaler.fit_transform(np.arctan(A21_input))[:, 0]
X[:,4] = scaler.fit_transform(np.arctan(A22_input))[:, 0]
X[:,5] = scaler.fit_transform(np.arctan(A31_input))[:, 0]
X[:,6] = scaler.fit_transform(np.arctan(A33_input))[:, 0]


input1 = 1/ (s11_DNS ** 3 * tau_DNS ** 2 - s11_DNS ** 2 * s22_DNS * tau_DNS ** 2 + 4 * s11_DNS * s12_DNS ** 2 * tau_DNS ** 2 - s11_DNS * s22_DNS ** 2 * tau_DNS ** 2 + 4 * s12_DNS ** 2 * s22_DNS * tau_DNS ** 2 + s22_DNS ** 3 * tau_DNS ** 2)
input2 = 1/ (s11_DNS ** 2 * tau_DNS ** 2 * w12_DNS - 2 * s11_DNS * s22_DNS * tau_DNS ** 2 * w12_DNS + 4 * s12_DNS ** 2 * tau_DNS ** 2 * w12_DNS + s22_DNS ** 2 * tau_DNS ** 2 * w12_DNS)
input3 = 1/ (s11_DNS ** 3 * tau_DNS ** 2 * w12_DNS ** 2 - s11_DNS ** 2 * s22_DNS * tau_DNS ** 2 * w12_DNS ** 2 + 4 * s11_DNS * s12_DNS ** 2 * tau_DNS ** 2 * w12_DNS ** 2 - s11_DNS * s22_DNS ** 2 * tau_DNS ** 2 * w12_DNS ** 2 + 4 * s12_DNS ** 2 * s22_DNS * tau_DNS ** 2 * w12_DNS ** 2 + s22_DNS ** 3 * tau_DNS ** 2 * w12_DNS ** 2)

input1 = input1.reshape(-1,1)
input2 = input2.reshape(-1,1)
input3 = input3.reshape(-1,1)

scaler = MinMaxScaler()
'''
X=np.zeros((len(dudy_DNS), 3))
X[:,0] = scaler.fit_transform(np.arctan(input1))[:, 0]
X[:,1] = scaler.fit_transform(np.arctan(input2))[:, 0]
X[:,2] = scaler.fit_transform(np.arctan(input3))[:, 0]'''

s11_input = (tau_DNS*s11_DNS).reshape(-1,1)
s12_input = (tau_DNS*s12_DNS).reshape(-1,1)
s21_input = (tau_DNS*s21_DNS).reshape(-1,1)
s22_input = (tau_DNS*s22_DNS).reshape(-1,1)
s33_input = (tau_DNS*s33_DNS).reshape(-1,1)

scaler = MinMaxScaler()
X2=np.zeros((len(dudy_DNS), 5))
X2[:,0] = scaler.fit_transform(s11_input)[:, 0]
X2[:,1] = scaler.fit_transform(s12_input)[:, 0]
X2[:,2] = scaler.fit_transform(s21_input)[:, 0]
X2[:,3] = scaler.fit_transform(s22_input)[:, 0]
X2[:,4] = scaler.fit_transform(s33_input)[:, 0]

random_state = randrange(100)

indices = np.arange(len(X))
X2_train, X2_test, y2_train, y2_test, index_train2, index_test2 = train_test_split(X2, y, indices, test_size=0.2, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indices, test_size=0.2, shuffle=True, random_state=42)



#-------------------Data split----------------------
c_DNS_train_dict = {'c1':c_DNS[0,index_train], 'c2':c_DNS[1,index_train], 'c3':c_DNS[2,index_train]}
c_DNS_test_dict = {'c1':c_DNS[0,index_test], 'c2':c_DNS[1,index_test], 'c3':c_DNS[2,index_test]}
c_DNS_test_list = c_DNS[:, index_test]
c_DNS_train_list = c_DNS[:, index_train]

y_DNS_test = y_DNS[index_test]
x_DNS_test = x_DNS[index_test]

uu_DNS_test = uu_DNS[index_test]
vv_DNS_test = vv_DNS[index_test]
ww_DNS_test = ww_DNS[index_test]
uv_DNS_test = uv_DNS[index_test]

dudx_DNS_test = dudx_DNS[index_test]
dudy_DNS_test = dudy_DNS[index_test]
dvdx_DNS_test = dvdx_DNS[index_test]
dvdy_DNS_test = dvdy_DNS[index_test]

eps_DNS_test = eps_DNS[index_test]
k_DNS_test = k_DNS[index_test]
tau_DNS_test = tau_DNS[index_test]

s11_DNS_test = s11_DNS[index_test]
s12_DNS_test = s12_DNS[index_test]
s21_DNS_test = s21_DNS[index_test]
s22_DNS_test = s22_DNS[index_test]
s33_DNS_test = s33_DNS[index_test]

w12_DNS_test = w12_DNS[index_test]
w21_DNS_test = w21_DNS[index_test]

a11_DNS_test = a11_DNS[index_test]
a12_DNS_test = a12_DNS[index_test]
a22_DNS_test = a22_DNS[index_test]
a33_DNS_test = a33_DNS[index_test]

c1_DNS_test = c1_DNS[index_test]
c2_DNS_test = c2_DNS[index_test]
c3_DNS_test = c3_DNS[index_test]

y_DNS_train = y_DNS[index_train]
x_DNS_train = x_DNS[index_train]

uu_DNS_train = uu_DNS[index_train]
vv_DNS_train = vv_DNS[index_train]
ww_DNS_train = ww_DNS[index_train]
uv_DNS_train = uv_DNS[index_train]

dudx_DNS_train = dudx_DNS[index_train]
dudy_DNS_train = dudy_DNS[index_train]
dvdx_DNS_train = dvdx_DNS[index_train]
dvdy_DNS_train = dvdy_DNS[index_train]

eps_DNS_train = eps_DNS[index_train]
k_DNS_train = k_DNS[index_train]
tau_DNS_train = tau_DNS[index_train]

s11_DNS_train = s11_DNS[index_train]
s12_DNS_train = s12_DNS[index_train]
s21_DNS_train = s21_DNS[index_train]
s22_DNS_train = s22_DNS[index_train]
s33_DNS_train = s33_DNS[index_train]

w12_DNS_train = w12_DNS[index_train]
w21_DNS_train = w21_DNS[index_train]

a11_DNS_train = a11_DNS[index_train]
a12_DNS_train = a12_DNS[index_train]
a22_DNS_train = a22_DNS[index_train]
a33_DNS_train = a33_DNS[index_train]

c1_DNS_train = c1_DNS[index_train]
c2_DNS_train = c2_DNS[index_train]
c3_DNS_train = c3_DNS[index_train]