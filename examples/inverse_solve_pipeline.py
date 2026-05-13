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
import yaml
import os

R0_design = 0.45


def init_psi():
	'''
	with inverse solve method, it will easy to fail as the init psi does not converge,
	here we apply the precalulated eqdsk 'first.geqdsk' as the initial guess to help the convergence.
	'''
	from freeqdsk import geqdsk
	f = open('first.geqdsk','r')
	#f = open('./data_first/base.geqdsk','r')
	data = geqdsk.read(f)
	psi = data['psi']
	return psi

def inverse_solve(config):
	'''
	with inverse solve, one is able to choose
	Ip: plasma current
	Paxis: the Pressure in Pa at magnetics axis
	am/an: parameters to define current and pressure profile within plasam
		   am:3, an:2 can be regarded as Ohmic profile.
	fvec = R*Btor (major radius, toroidal B field)
		   in first, R = 0.45, Btor = 0.06, 0.1, etc.
	plasma shape paramters:
	 ellip = 2.0
	 tria = 0.5
	 R0 = 0.45
	 fac = 0.9
	one can play with plasma_boundary() to see what kind of plasma shpape you are interested.
	
	when the plasm shape has been defined, it will be put into solver as optimization constraint.
	However, the function produce way to many points, therefore in will be reduce 20X before sending to solver.

	in inverse solve, we defined target: Ip, Paxis, plasma shape, 
	and solve for PFC and CS currents that will achieve those targets.	
	
	in GSStaticSolver.inverse_solve() , the parameter 'l2_reg=[1e-9]*5' stands for how many independent PFC coils
	one would to optimized to. 
	for example:
		for array with size 5, we have [PF1, PF3, PF4, PF5, CS]
		for array with size 3, we might have [PF_13, PF_45, CS]
	here we have been ignoring PF2 and PF6 from the original design so far, as they 
	1. for active vertical control
	2. not help with convergence
	
	however, if one would like to add them back, one should set up their coil settings in first_machine.py 
	and setting l2_reg = [1e-6,1e-6, 1e-9,1e-9,1e-9] for  [PF2_up, PF2_down, PF_13, PF_45, CS], for instance.
	PF2 if for vertical control, so they can not have symmetric configuration, and the l2_reg is set bigger than other PFCs.

	one can set  
	eq.tokamak['Solenoid'].control = False
	to fix CS coil current through out optimization   

	By default, the solution is saved as *.geadsk file, and the PFC coils as *pk file
	in data_first 
	
	there is a plotting tool to parse *geqdsk file, located in	
	freegsnke_v2.1.0_nchc/tools
	 
	'''
	Rmin = config['Rmin']
	Rmax = config['Rmax']
	Zmin = config['Zmin']
	Zmax = config['Zmax']
	nx = config['nx']
	ny = config['ny']

	tokamak_first = build_first()
	plasma_psi = init_psi()
	eq = equilibrium_update.Equilibrium(
		tokamak=tokamak_first,
		Rmin = Rmin, Rmax = Rmax, 
		Zmin = Zmin, Zmax = Zmax,
		nx = nx,
		ny = ny,
		psi = plasma_psi
	)
	
	Ip = config['Ip']
	paxis = config['paxis']
	alpha_m	= config['alpha_m']
	alpha_n	= config['alpha_n']
	Btor  = config['Btor']	
	
	fvec = Btor*R0_design
	profiles = ConstrainPaxisIp(
		eq=eq,
		paxis=paxis,
		Ip=Ip,
		fvac=fvec,
		alpha_m=alpha_m,
		alpha_n=alpha_n
	)



	GSStaticSolver = GSstaticsolver.NKGSsolver(eq)
	
	ellip = config['ellip']
	tria = config['tria']
	Raxis = config['Raxis']
	fac = config['fac']
	iso_R, iso_Z = plasma_boundary(factor=fac, ellip=ellip, tria=tria, R0=Raxis)	
	lenR = len(iso_R)	
	iso_sub_R = []
	iso_sub_Z = []
	for idx, (R, Z) in enumerate(zip(iso_R, iso_Z)):	
		if idx%20 ==0:
			iso_sub_R.append(R)
			iso_sub_Z.append(Z)

	isoflux_set = np.array([[iso_sub_R, iso_sub_Z]])
	print('isoflux_set {}'.format(isoflux_set))
	#constrain = Inverse_optimizer(null_points=null_points,isoflux_set=isoflux_set)	
	constrain = Inverse_optimizer(isoflux_set=isoflux_set)	

	eq.tokamak.set_coil_current('Solenoid', 0.2)
	eq.tokamak['Solenoid'].control = True
	converged = GSStaticSolver.inverse_solve(eq=eq, 
						 profiles=profiles, 
						 constrain=constrain,
						 target_relative_tolerance=1e-6,
						 target_relative_psit_update=1e-3,
						 verbose=True, 
						 #l2_reg=[1e-9]*3
						 l2_reg=[1e-9]*5
						)
	print('convergence : {}'.format(converged))
	# save to data if converges
	if converged:
		inverse_current_values = eq.tokamak.getCurrents()
	
		opt = eq._profiles.opt
		xpt = eq._profiles.xpt
		oxpoints = (opt, xpt)
		from datetime import datetime
		date_str = datetime.today().strftime('%Y_%m_%d')
		exp_name = 'exp_{}_Ip{}_Paxis{}_Btor{}_am{}an{}_R0{}_fac{}_ellip{}_tria{}'.format(date_str, Ip, paxis, Btor, alpha_m, alpha_n, Raxis,  fac ,ellip,tria)
		os.makedirs(os.path.join('./data_first',exp_name),exist_ok=True)

		with open('data_first/{}/current_{}.pk'.format(exp_name,exp_name), 'wb') as f:
			pickle.dump(obj=inverse_current_values, file=f)
		savefile = 'data_first/{}/eqdsk_{}.geqdsk'.format(exp_name,exp_name)
		with open(savefile, "w") as f:
			geqdsk.write(eq,f, oxpoints=oxpoints)
		config_out = 'data_first/{}/config_{}.yaml'.format(exp_name, exp_name)
		with open(config_out,'w') as fp:
			yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
		fp.close()
		# extract measuremensts
		tokamak_first.probes.initialise_setup(eq)	
		probe_list = tokamak_first.probes.floops + tokamak_first.probes.pickups
		probe_names = []
		for probe in probe_list:
			probe_names.append(probe['name'])
		probe_values = np.zeros(len(probe_names))
		probe_values = np.concatenate([tokamak_first.probes.calculate_fluxloop_value(eq)*2*np.pi, # multiply by 2pi to match flux units in UDA
        						   tokamak_first.probes.calculate_pickup_value(eq)
    							  ])
		mrms_dict = {}
		for name, val in zip(probe_names,probe_values):
			mrms_dict[name] = val
		
		mrms_out = 'data_first/{}/mrms_{}.pk'.format(exp_name,exp_name)
		with open(mrms_out,'wb') as f:
			pickle.dump(obj=mrms_dict, file=f)
	
		print(mrms_dict)


	plot_eq(eq, constrain)

def plot_eq(eq, constrain):
	fig1, ax1 = plt.subplots(1, 1, figsize=(4, 8), dpi=80)
	ax1.grid(True, which='both')
	eq.plot(axis=ax1, show=False)
	eq.tokamak.plot(axis=ax1, show=False)
	constrain.plot(axis=ax1,show=True)
	#ax1.set_xlim(0.01, 2.5)
	#ax1.set_ylim(-3,3)
	#plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	with open('config.yaml','r') as f:
		config_dict = yaml.safe_load(f)
	
	inverse_solve(config_dict)


