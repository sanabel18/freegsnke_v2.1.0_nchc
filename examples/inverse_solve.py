import numpy as np
from freegsnke import equilibrium_update
from freegsnke.jtor_update import ConstrainPaxisIp
from freegsnke import GSstaticsolver
from machine_first import build_first
from machine_first import plasma_boundary
from freegsnke.inverse import Inverse_optimizer
import matplotlib.pyplot as plt
import pickle
from freegs4e import geqdsk

def init_psi():
	from freeqdsk import geqdsk
	f = open('/Users/hsiaohsl/workspace/pkg/sana_narlabs/tmp/freegsnke_v2.1.0_nchc/examples/data_first/eqdsk_Ip_30000.0_Paxis_32_am2an_3_R00.45_fac0.9_ellip2.2_tria0.5.geqdsk','r')
	#f = open('./data_first/base.geqdsk','r')
	data = geqdsk.read(f)
	psi = data['psi']
	return psi

def inverse_solve():
	tokamak_first = build_first()
	plasma_psi = init_psi()
	eq = equilibrium_update.Equilibrium(
		tokamak=tokamak_first,
		Rmin = 0.01, Rmax = 2.5, 
		Zmin = -3, Zmax = 3,
		nx = 257,
		ny = 257,
		psi = plasma_psi
	)
	Ip = 3e4
	paxis=32
	am = 2
	an = 3
	profiles = ConstrainPaxisIp(
		eq=eq,
		paxis=paxis,
		Ip=Ip,
		fvac=0.027,
		alpha_m=am,
		alpha_n=an
	)



	GSStaticSolver = GSstaticsolver.NKGSsolver(eq)
	
	#iso_R, iso_Z = plasma_boundary(factor=0.9, ellip=2.2, tria=0.5)	
	ellip = 2.0
	tria = 0.8
	R0 = 0.46
	fac = 0.9
	iso_R, iso_Z = plasma_boundary(factor=fac, ellip=ellip, tria=tria, R0=R0)	
	lenR = len(iso_R)	
	iso_sub_R = []
	iso_sub_Z = []
	for idx, (R, Z) in enumerate(zip(iso_R, iso_Z)):	
		if idx%20 ==0:
			iso_sub_R.append(R)
			iso_sub_Z.append(Z)

	#isoflux_set = np.array([[iso_R.tolist(), iso_Z.tolist()]])
	isoflux_set = np.array([[iso_sub_R, iso_sub_Z]])
	print('isoflux_set {}'.format(isoflux_set))
	#null_points = [[0.48],[0.0]]
	#constrain = Inverse_optimizer(null_points=null_points,isoflux_set=isoflux_set)	
	constrain = Inverse_optimizer(isoflux_set=isoflux_set)	

	eq.tokamak.set_coil_current('Solenoid', 0.2)
	eq.tokamak['Solenoid'].control = True
	GSStaticSolver.inverse_solve(eq=eq, 
						 profiles=profiles, 
						 constrain=constrain,
						 target_relative_tolerance=1e-6,
						 target_relative_psit_update=1e-3,
						 verbose=True, 
						 #l2_reg=[1e-9]*3
						 l2_reg=[1e-9]*5
						)
	inverse_current_values = eq.tokamak.getCurrents()
	#print(inverse_current_values)
	
	#plot_eq(eq, constrain)
	opt = eq._profiles.opt
	xpt = eq._profiles.xpt
	oxpoints = (opt, xpt)

	with open('data_first/current_Ip_{}_Paxis_{}_am{}an_{}_R0{}_fac{}_ellip{}_tria{}.pk'.format(Ip, paxis, am, an, R0,  fac ,ellip,tria), 'wb') as f:
		pickle.dump(obj=inverse_current_values, file=f)
	#plot_eq(eq, constrain)
	savefile = 'data_first/eqdsk_Ip_{}_Paxis_{}_am{}an_{}_R0{}_fac{}_ellip{}_tria{}.geqdsk'.format(Ip, paxis, am, an, R0, fac ,ellip,tria)
	with open(savefile, "w") as f:
		geqdsk.write(eq,f, oxpoints=oxpoints)


	# save coil currents to file
	#with open('data_first/first_test_currents_PaxisIp.pk', 'wb') as f:
	#	pickle.dump(obj=inverse_current_values, file=f)	
	plot_eq(eq, constrain)

def plot_eq(eq, constrain):
	fig1, ax1 = plt.subplots(1, 1, figsize=(4, 8), dpi=80)
	ax1.grid(True, which='both')
	eq.plot(axis=ax1, show=False)
	eq.tokamak.plot(axis=ax1, show=False)
	constrain.plot(axis=ax1,show=True)
	ax1.set_xlim(0.01, 2.5)
	ax1.set_ylim(-3,3)
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	inverse_solve()


