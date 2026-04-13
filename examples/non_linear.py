import numpy as np
from freegsnke import equilibrium_update
from freegsnke.jtor_update import ConstrainPaxisIp
from freegsnke import GSstaticsolver
from machine_first import build_first
from machine_first import plasma_boundary
from freegsnke.inverse import Inverse_optimizer
import matplotlib.pyplot as plt
import pickle
from inverse_solve import init_psi
from freegsnke import nonlinear_solve



def nonlinear():
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
		 
	profiles = ConstrainPaxisIp(
		eq=eq,
		paxis=32,
		Ip=3e4,
		fvac=0.027,
		alpha_m=2,
		alpha_n=3
	)

	GSStaticSolver = GSstaticsolver.NKGSsolver(eq)
	
	with open('data_first/first_test_currents_PaxisIp.pk', 'rb') as f:
		currents_dict = pickle.load(f)
	
	# assign currents to the eq object
	for key in currents_dict.keys():
		eq.tokamak.set_coil_current(coil_label=key, current_value=currents_dict[key])
	
	current = eq.tokamak.getCurrents()	
	print(len(current))
	GSStaticSolver.solve(eq=eq,
						profiles=profiles,
						constrain=None,
						target_relative_tolerance=1e-9
						)
	#fig1, ax1 = plt.subplots(1, 1, figsize=(4, 8), dpi=60)
	#ax1.grid(True, which='both')
	#eq.plot(axis=ax1, show=False)
	#eq.tokamak.plot(axis=ax1, show=False)
	#ax1.set_xlim(0.10, 2.5)
	#ax1.set_ylim(-3, 3)
	#plt.tight_layout()
	nonlinear_solver = nonlinear_solve.nl_solver(
    	eq=eq, 
    	profiles=profiles, 
    	GSStaticSolver=GSStaticSolver, 
    	plasma_resistivity=1e-6, # this defines the lumped plasma resistances
    	min_dIy_dI=0,            # this has been set artificially low in this example
    	threshold_dIy_dI=1,      # this has been set artificially high in this example
    	max_mode_frequency=1e2  # this has been set artificially high in this example
		)
	print(f"Total number of modes excl. plasma current = {nonlinear_solver.n_metal_modes}")   # total (actives + passives)
	print(f"Total number of active coils = {nonlinear_solver.n_active_coils}")   # actives
	print(f"Total number of passive structures = {nonlinear_solver.n_passive_coils}")   # passives


if __name__ == '__main__':
	nonlinear()


