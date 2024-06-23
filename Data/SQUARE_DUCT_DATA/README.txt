Direct numerical simulation dataset of turbulent square duct flow up to friction Reynolds number Retau=1000
Reference publication:

D. Modesti Theor. Comput. Fluid Dyn. 2020

Other references:

S. Pirozzoli, D. Modesti, P. Orlandi, F. Grasso, 2018, J. Fluid Mech. 840, 631-655  https://doi.org/10.1017/jfm.2018.66

D. Modesti, S. Pirozzoli, P. Orlandi, F. Grasso, 2018, J. Fluid Mech, 847, R1  https://doi.org/10.1017/jfm.2018.391

The present dataset include Turbulent square duct flow data at friction Reynolds numbers Re_tau=150,227,519,1055

Mean flow fields are in Tecplot ASCII format and contain the following variables:
*
utau is the mean friction velocity, averaged over the duct perimeter
deltav is the mean viscous length scale, averaged over the duct perimeter
*
stress_n.dat, n is the number of basis used in Pope's polynomial expansion

1) y/h
2) z/h
3) u/utau   ! velocity components 
4) v/utau 
5) w/utau 
6) s12/utau ! strain of rate tensor
7) s13/utau 
8) s22/utau 
9) s23/utau 
10) s33/utau 
10) s33/utau 
11) a11/utau^2 ! anisotropic Reynolds stress tensor from DNS
12) a12/utau^2
13) a13/utau^2
14) a22/utau^2
15) a23/utau^2
16) a33/utau^2
17) amod11/utau^2 ! modelled anisotropic Reynolds stress tensor (equation 5 in reference publication)
18) amod12/utau^2
19) amod13/utau^2
20) amod22/utau^2
21) amod23/utau^2
22) amod33/utau^2
23) eps*deltav/utau^3 ! dissipation from DNS 
24) tke/utau^2        ! turbulent kinetic energy from DNS 
25) G1/(utau*deltav)  ! Coefficients of Pope's polynomial expansion (equation 8 in reference publication) 
26) G2/deltav^2                
27) G3/deltav^2                
28) G4/deltav^2                
29) G5/(deltav^3/utau)                
30) utau              ! friction velocity 
31) deltav            ! friction velocity 
*

tbas.dat contains the tensor bases of Pope's polynomial expansion computed using DNS data

1) T1(1,1)/(utau/deltav)
2) T1(1,2)/(utau/deltav)
3) T1(1,3)/(utau/deltav)
4) T1(2,2)/(utau/deltav)
5) T1(2,3)/(utau/deltav)
6) T1(3,3)/(utau/deltav)
7) T2(1,1)/(utau/deltav)**2
8) T2(1,2)/(utau/deltav)**2
9) T2(1,3)/(utau/deltav)**2
10) T2(2,2)/(utau/deltav)**2
11) T2(2,3)/(utau/deltav)**2
12) T2(3,3)/(utau/deltav)**2
13) T3(1,1)/(utau/deltav)**2
14) T3(1,2)/(utau/deltav)**2
15) T3(1,3)/(utau/deltav)**2
16) T3(2,2)/(utau/deltav)**2
17) T3(2,3)/(utau/deltav)**2
18) T3(3,3)/(utau/deltav)**2
19) T4(1,1)/(utau/deltav)**2
20) T4(1,2)/(utau/deltav)**2
21) T4(1,3)/(utau/deltav)**2
22) T4(2,2)/(utau/deltav)**2
23) T4(2,3)/(utau/deltav)**2
24) T4(3,3)/(utau/deltav)**2
25) T5(1,1)/(utau/deltav)**3
26) T5(1,2)/(utau/deltav)**3
27) T5(1,3)/(utau/deltav)**3
28) T5(2,2)/(utau/deltav)**3
29) T5(2,3)/(utau/deltav)**3
30) T5(3,3)/(utau/deltav)**3
