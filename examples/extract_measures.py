import numpy as np
from freegsnke import GSstaticsolver
from freegsnke import equilibrium_update
from freegsnke.jtor_update import ConstrainPaxisIp
from machine_first import build_first
from freegsnke.inverse import Inverse_optimizer
import matplotlib.pyplot as plt
import pickle
from freegs4e import geqdsk
import yaml

R0 = 0.45


def extract_measures(config, current_in):
	tokamak_first = build_first()
	
	Rmin = config['Rmin']
	Rmax = config['Rmax']
	Zmin = config['Zmin']
	Zmax = config['Zmax']
	nx = config['nx']
	ny = config['ny']
	
	eq = equilibrium_update.Equilibrium(
		tokamak=tokamak_first,
		Rmin = Rmin, Rmax = Rmax, 
		Zmin = Zmin, Zmax = Zmax,
		nx = nx,
		ny = ny,
		#psi = plasma_psi
	)
	
	paxis = config['paxis']
	Ip	= config['Ip']
	fvac = R0*config['Btor']
	alpha_m = config['alpha_m']
	alpha_n = config['alpha_n']
		 
	profiles = ConstrainPaxisIp(
		eq=eq,
		paxis=paxis,
		Ip=Ip,
		fvac=fvac,
		alpha_m=alpha_m,
		alpha_n=alpha_n
	)

	GSStaticSolver = GSstaticsolver.NKGSsolver(eq)
		
	# assign currents to the eq object
	for key in current_in.keys():
		eq.tokamak.set_coil_current(coil_label=key, current_value=current_in[key])
	
	current = eq.tokamak.getCurrents()	
	GSStaticSolver.solve(eq=eq,
						profiles=profiles,
						constrain=None,
						target_relative_tolerance=1e-9
						)

	tokamak_first.probes.initialise_setup(eq)	
	probe_list = tokamak_first.probes.floops + tokamak_first.probes.pickups
	probe_names = []
	for probe in probe_list:
		probe_names.append(probe['name'])
	probe_values = np.zeros(len(probe_names))
	probe_values = np.concatenate([tokamak_first.probes.calculate_fluxloop_value(eq)*2*np.pi, # multiply by 2pi to match flux units in UDA
        						   tokamak_first.probes.calculate_pickup_value(eq)
    							  ])
	print('probe_names {}'.format(probe_names))
	print('probe_values {}'.format(probe_values))

def create_yaml_config(outfile):
	config = {}
	config['Rmin'] = 0.01
	config['Rmax'] = 2.5
	config['Zmin'] = -3
	config['Zmax'] = 3
	config['nx'] = 257
	config['ny'] = 257
	config['paxis'] = 32
	config['Ip'] = 3e4
	config['Btor'] = 0.06
	config['alpha_m'] = 2
	config['alpha_n'] =3
	with open(outfile,'w') as fp:
		yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
	fp.close()

if __name__ == '__main__':
	create_yaml_config('config_template.yaml')
	
	with open('config_template.yaml','r') as f:
		config_dict = yaml.safe_load(f)
	
	current_file = './data_first/current_Ip_30000.0_Paxis_32_am2an_3_R00.45_fac0.9_ellip2.2_tria0.5.pk'	
	with open(current_file, 'rb') as f:
		currents_dict = pickle.load(f)

	extract_measures(config_dict, currents_dict)






