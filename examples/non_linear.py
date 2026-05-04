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

from freegs4e import geqdsk

def export_geadsk():
	tokamak_first = build_first()
	plasma_psi = init_psi()
	eq = equilibrium_update.Equilibrium(
		tokamak=tokamak_first,
		Rmin = 0.01, Rmax = 1, 
		Zmin = -1, Zmax = 1,
		nx = 65,
		ny = 65,
		#psi = plasma_psi
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
	#

	opt = eq._profiles.opt
	xpt = eq._profiles.xpt
	oxpoints = (opt, xpt)

	with open("FIRST_13_45_3e4_32_small_reso.geqdsk", "w") as f:
		 geqdsk.write(eq, f, oxpoints=oxpoints)
	



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
    	plasma_resistivity=7.8e-7, # this defines the lumped plasma resistances
    	min_dIy_dI=0.01,            # this has been set artificially low in this example
    	threshold_dIy_dI=0.025,      # this has been set artificially high in this example
    	max_mode_frequency=1e2  # this has been set artificially high in this example
		)
	print(f"Total number of modes excl. plasma current = {nonlinear_solver.n_metal_modes}")   # total (actives + passives)
	print(f"Total number of active coils = {nonlinear_solver.n_active_coils}")   # actives
	print(f"Total number of passive structures = {nonlinear_solver.n_passive_coils}")   # passives

	timescales = nonlinear_solver.linearised_sol.all_timescales # all eigenvalues: timescales
	growth_rates = 1/timescales                                 # growth rates are simply 1/timescales
	modes = nonlinear_solver.linearised_sol.all_modes           
	print('timescales {}'.format(timescales))
	print('modes: {}'.format(modes))
	mask = (timescales > 0)
	idx = np.where(mask)[0][0] # index of unstable mode
	unstable_timescales = timescales[mask]
	unstable_modes = np.squeeze(modes[:,mask])
	
	i = idx # default is unstable mode
	mode_currents = np.real(modes[:,i])
	print(f"Mode {i} ---> {'stable' if np.real(timescales[i]) < 0 else 'unstable'}")
	print(f"Growth rate = {np.real(growth_rates[i]):.2e} [1/s]")
	print(f"Timescale = {np.real(timescales[i]):.2e} [s]")




if __name__ == '__main__':
	nonlinear()
	#export_geadsk()

