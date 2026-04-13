import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as pts
from pathlib import Path

# the PF coil are build with cooper wire with 9mm and 7mm radius
# R = rho*L/A

area = np.pi*(9e-3**2- 7e-3**2)
sqa = np.sqrt(area)
#L = np.pi*0.6**2*20
#copper_rho = R*area/L
resistivity_steel = 7.2e-7 # 304 steel
eta_copper = 1.55e-8  # resistivity in Ohm*m

def calc_windings(R, Z, dR, dZ, wire_R, wire_Z):
	# PF coil wired with wire_R X wire_Z
	# turns = wire_R*wire_Z
	R_start = R - 0.5*dR
	R_end = R + 0.5*dR
	Z_start = Z - 0.5*dZ
	Z_end = Z + 0.5*dZ
	pos_R = np.linspace(R_start, R_end, wire_R, endpoint=True)
	pos_Z = np.linspace(Z_start, Z_end, wire_Z, endpoint=True)
	rv, zv = np.meshgrid(pos_R, pos_Z)
	rv = rv.reshape((1, wire_R*wire_Z))
	zv = zv.reshape((1, wire_R*wire_Z))
	#fig, ax = plt.subplots(1,1,figsize=(4,5))
	#ax.scatter(rv, zv)
	#ax.set_aspect('equal')
	#plt.show()
	return rv[0], zv[0]

def solenoid():
	# inner coil: 173/2mm
	# outter coil: 186/2 mm
    # 5mm X 8mm with 2mm hole
	# area = 0.005*0.008-np.pi*0.002*0.002
    # effective radius = np.sqrt(0.005*0.008-np.pi*0.002*0.002)	
	R = [0.173/2]*170 + [0.186/2]*170
	Z = np.linspace(-0.8, 0.8, 170, endpoint=True)
	Z = np.concatenate((Z,Z), axis=0)
	
	radius = np.sqrt(0.005*0.008-np.pi*0.002*0.002)
	return R, Z, radius

	

def set_coil_entry(active_coils, R, Z, length, key1, key2=None):
	if key2:
		active_coils[key1][key2] = {
			"R": R,
			"Z": Z,
			"dR": length,
			"dZ": length,
			"resistivity": eta_copper,
			"polarity": 1,
			"multiplier": 1,
		}
	else:
		active_coils[key1] = {
			"R": R,
			"Z": Z,
			"dR": length,
			"dZ": length,
			"resistivity": eta_copper,
			"polarity": 1,
			"multiplier": 1,
		}
		
	return active_coils

def set_passive_coil_entry( R, Z, dR, dZ):
	pass_dict = {
		"R": R,
		"Z": Z,
		"dR": dR,
		"dZ": dZ,
		"resistivity": resistivity_steel,
	}
	return pass_dict

def active_coils_13_45(): 
	active_coils = {}
	
	active_coils['PF13'] = {}	
	PF1N_R, PF1N_Z = 	calc_windings(0.95, 0.2, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF1N_R, PF1N_Z, sqa, 'PF13', 'upper_1')
	PF1S_R, PF1S_Z = 	calc_windings(0.95, -0.2, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF1S_R, PF1S_Z, sqa, 'PF13', 'lower_1')
	
	PF3N_R, PF3N_Z = 	calc_windings(0.86, 0.63, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF3N_R, PF3N_Z, sqa, 'PF13', 'upper_3')
	PF3S_R, PF3S_Z = 	calc_windings(0.86, -0.63, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF3S_R, PF3S_Z, sqa, 'PF13', 'lower_3')
	
	active_coils['PF45'] = {}	
	PF4N_R, PF4N_Z = 	calc_windings(0.55, 0.92, 0.052, 0.065, 4, 5)
	active_coils = set_coil_entry(active_coils, PF4N_R, PF4N_Z, sqa, 'PF45', 'upper_4')
	PF4S_R, PF4S_Z = 	calc_windings(0.55, -0.92, 0.052, 0.065, 4, 5)
	active_coils = set_coil_entry(active_coils, PF4S_R, PF4S_Z, sqa, 'PF45', 'lower_4')
	
	PF5N_R, PF5N_Z = 	calc_windings(0.16, 0.93, 0.05031, 0.052, 4, 4)
	active_coils = set_coil_entry(active_coils, PF5N_R, PF5N_Z, sqa, 'PF45', 'upper_5')
	PF5S_R, PF5S_Z = 	calc_windings(0.16, -0.93, 0.05031, 0.052, 4, 4)
	active_coils = set_coil_entry(active_coils, PF5S_R, PF5S_Z, sqa, 'PF45', 'lower_5')
	
	# Solenoid
	
	R_CS, Z_CS, r_CS = solenoid()
	active_coils = set_coil_entry(active_coils, R_CS, Z_CS, r_CS, 'Solenoid')

	return active_coils




def active_coils(): 
	active_coils = {}
	
	active_coils['PF1'] = {}	
	PF1N_R, PF1N_Z = 	calc_windings(0.95, 0.2, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF1N_R, PF1N_Z, sqa, 'PF1', 'upper')
	PF1S_R, PF1S_Z = 	calc_windings(0.95, -0.2, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF1S_R, PF1S_Z, sqa, 'PF1', 'lower')
	
	active_coils['PF3'] = {}	
	PF3N_R, PF3N_Z = 	calc_windings(0.86, 0.63, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF3N_R, PF3N_Z, sqa, 'PF3', 'upper')
	PF3S_R, PF3S_Z = 	calc_windings(0.86, -0.63, 0.065, 0.052, 5, 4)
	active_coils = set_coil_entry(active_coils, PF3S_R, PF3S_Z, sqa, 'PF3', 'lower')
	
	active_coils['PF4'] = {}	
	PF4N_R, PF4N_Z = 	calc_windings(0.55, 0.92, 0.052, 0.065, 4, 5)
	active_coils = set_coil_entry(active_coils, PF4N_R, PF4N_Z, sqa, 'PF4', 'upper')
	PF4S_R, PF4S_Z = 	calc_windings(0.55, -0.92, 0.052, 0.065, 4, 5)
	active_coils = set_coil_entry(active_coils, PF4S_R, PF4S_Z, sqa, 'PF4', 'lower')
	
	active_coils['PF5'] = {}	
	PF5N_R, PF5N_Z = 	calc_windings(0.16, 0.93, 0.05031, 0.052, 4, 4)
	active_coils = set_coil_entry(active_coils, PF5N_R, PF5N_Z, sqa, 'PF5', 'upper')
	PF5S_R, PF5S_Z = 	calc_windings(0.16, -0.93, 0.05031, 0.052, 4, 4)
	active_coils = set_coil_entry(active_coils, PF5S_R, PF5S_Z, sqa, 'PF5', 'lower')
	
	# Solenoid
	
	R_CS, Z_CS, r_CS = solenoid()
	active_coils = set_coil_entry(active_coils, R_CS, Z_CS, r_CS, 'Solenoid')

	return active_coils


def passive_coils():
	"""
	here we use filament to model our metal chamber
	center cylindar part [2mm blocks]
	outer arc part [10mm blocks]	
	"""
	# ARC
	R_arc = 968
	R_off = 140
	R_top = 540*0.5 
	
	theta_start = np.arccos((R_off + R_top)/R_arc)
	Z_start = R_arc*np.sin(theta_start)
	num_block = int(Z_start*2//10)
	Z_pos_arc = np.array([Z_start-10*i for i in range(num_block)])
	Z_pos_arc = Z_pos_arc - 5
	R_pos_arc = np.sqrt(968*968 - Z_pos_arc*Z_pos_arc) - 140
	print('passive num {}'.format(num_block))	

	Z_inner_start = 1675/2
	num_inner_block = int(Z_inner_start*2//20)
	Z_pos_inner = np.array([Z_inner_start-20*i for i in range(num_inner_block)])
	Z_pos_inner = Z_pos_inner - 10
	R_pos_inner = np.array([100]*num_inner_block)
	
	R_pos_inner = R_pos_inner/1000
	Z_pos_inner = Z_pos_inner/1000
	R_pos_arc = R_pos_arc/1000
	Z_pos_arc = Z_pos_arc/1000	

	# set passive coils
	passive_coils = []
	
	for R, Z in zip(R_pos_inner, Z_pos_inner):
		pass_dict = set_passive_coil_entry( R, Z, 0.002, 0.002)
		passive_coils.append(pass_dict)
	for R,Z in zip(R_pos_arc, Z_pos_arc):
		pass_dict = set_passive_coil_entry( R, Z, 0.01, 0.01)
		passive_coils.append(pass_dict)
	
	return passive_coils

	'''	
	fig, ax = plt.subplots(1,1,figsize=(4,5))
	for R, Z in zip(R_pos_arc, Z_pos_arc):
		rz = (R-5,Z-5)	
		rect = pts.Rectangle(rz,10,10)
		ax.add_patch(rect)
	
	for R, Z in zip(R_pos_inner, Z_pos_inner):
		rz = (R-1,Z-1)	
		rect = pts.Rectangle(rz,2,2)
		ax.add_patch(rect)
	ax.set_ylim(np.min(Z_pos_arc), np.max(Z_pos_arc))
	ax.set_xlim(0, np.max(R_pos_arc))	
	ax.set_aspect('equal', adjustable='box')
	plt.show()
	'''
def set_pickup_coil(name, pos, orient, orient_v):
	pick_c = {
		'name':					name,
		'position': 			pos,
		'orientation':			orient,
		'orientation_vector':   orient_v,
	}
	return pick_c
	
def bdots():
	
	#'''	
	start = -6
	num_bdot = 15
	n = start + np.arange(0,num_bdot)
	n_rev = n[::-1]
	angle = np.deg2rad(7.555555)
	bdot_offset_angle = np.tan(30/957.5)
	angle_radian = (n_rev - 1)*angle + bdot_offset_angle
	R_outer = 957.5*np.cos((n_rev - 1)*angle + bdot_offset_angle ) -140
	Z_outer = 957.5*np.sin((n_rev - 1)*angle + bdot_offset_angle)
	R_inner = np.array([120.04]*num_bdot) 
	Z_inner = np.array((n-1)*100 + 50)
	R_outer = R_outer/1000
	R_inner = R_inner/1000
	Z_outer = Z_outer/1000
	Z_inner = Z_inner/1000

	
	R_md = R_outer.tolist() + R_inner.tolist()
	#R_md = np.array(R_md)/1000.
	Z_md = Z_outer.tolist() + Z_inner.tolist()
	#Z_md = np.array(Z_md)/1000.
	#'''
	pickup_coils = []
	idx = 0
	for R, Z in zip(R_inner, Z_inner):
		name = 'bdot_inner_{:02d}_para'.format(idx)
		pos = np.array([R, 0.0, Z])
		orient = 'PARALLEL'
		orient_v = np.array([0.,0.,1.])
		pick_dict =  set_pickup_coil(name, pos, orient, orient_v)
		pickup_coils.append(pick_dict)
		name = 'bdot_inner_{:02d}_norm'.format(idx)
		pos = np.array([R, 0.0, Z])
		orient = 'NORMAL'
		orient_v = np.array([1.,0.,0.])
		pick_dict =  set_pickup_coil(name, pos, orient, orient_v)
		idx = idx + 1
		pickup_coils.append(pick_dict)
	
	for R, Z, angle in zip(R_outer, Z_outer, angle_radian):
		name = 'bdot_outer_{:02d}_para'.format(idx)
		pos = np.array([R, 0.0, Z])
		orient = 'PARALLEL'
		orient_v = np.array([-np.sin(angle) ,0., np.cos(angle)])
		pick_dict =  set_pickup_coil(name, pos, orient, orient_v)
		pickup_coils.append(pick_dict)
		name = 'bdot_outer_{:02d}_normal'.format(idx)
		pos = np.array([R, 0.0, Z])
		orient = 'NORMAL'
		orient_v = np.array([np.cos(angle) ,0., np.sin(angle)])
		pick_dict =  set_pickup_coil(name, pos, orient, orient_v)
		idx = idx + 1
		pickup_coils.append(pick_dict)
		
	return pickup_coils


def set_flux_loop(name, pos):
	flux_dict = {
		'name':     name,
		'position': pos
	}
	return flux_dict

def flux_loop():
	 
	start = -7
	num_flux = 8-(-7) + 1
	n = start + np.arange(0,num_flux)
	n_rev = n[::-1]
	angle = np.deg2rad(7.555555)
	R_outer = 960*np.cos((n_rev - 0.5)*angle) -140
	Z_outer = 960*np.sin((n_rev - 0.5)*angle)
	R_inner = np.array([117.04]*num_flux) 
	Z_inner = np.array((n)*100 -30)
	R_md = R_outer.tolist() + R_inner.tolist()
	R_md = np.array(R_md)/1000.
	Z_md = Z_outer.tolist() + Z_inner.tolist()
	Z_md = np.array(Z_md)/1000.
	RSI = R_md.tolist()
	ZSI = Z_md.tolist()
		
	flux_loops = []
	idx = 0
	for R, Z in zip(RSI, ZSI):
		name = 'flux_loop_{:2d}'.format(idx)
		pos = np.array([R,Z])
		flux_dict = set_flux_loop(name, pos )
		flux_loops.append(flux_dict)
		idx = idx + 1
	return flux_loops

def plasma_boundary(factor=1.2, ellip=2.4, tria=0.5, R0=0.45):
	tria_u = tria
	tria_l = tria
	quad_u = 0
	quad_l = 0
	#R0 = 0.45
	Z0 = 0.0
	#a0 = 0.319599393
	#a0 = 0.38
	a0 = 0.32
	a0 = factor*a0
	angle = np.linspace(0,2*np.pi,180)
	angle = angle[:-1]
	
	R = R0 + a0 * np.cos(angle + tria_u*np.sin(angle) + quad_u*np.sin(2*angle))
	Z = Z0 + a0 * ellip * np.sin(angle)
	#R = np.cos(theta + tri_u*np.sin(theta) + quad_u*np.sin(2*theta))
	#Z = ellip*np.sin(theta)
	return R, Z	

def set_limiter(R, Z):
	lim_dict = {
		"R": R,
		"Z": Z,
	}
	return lim_dict

def limiter(factor=1.0, ellip=2.2, tria=0.5):
	R_lim, Z_lim = plasma_boundary(factor=factor, ellip=ellip, tria=tria)
	lim = []
	for R, Z in zip(R_lim, Z_lim):
		lim_dict = set_limiter(R,Z)
		lim.append(lim_dict)
	return lim


		
def save_machine(active_coils, passive_coils, limiter, magnetic_probes,  save_path):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	
	save_path = Path(save_path)

	with open(save_path / 'active_coils.pickle', "wb") as f:
		pickle.dump(active_coils, f)
	with open(save_path / 'passive_coils.pickle', "wb") as f:
		pickle.dump(passive_coils, f)
	with open(save_path / 'limiter.pickle', "wb") as f:
		pickle.dump(limiter, f)
	with open(save_path / 'magnetic_probes.pickle', "wb") as f:
		pickle.dump(magnetic_probes, f)
			

def plot_machine(tokamak):
	fig1, ax1 = plt.subplots(1, 1, figsize=(4, 8), dpi=80)
	
	plt.tight_layout()

	tokamak.plot(axis=ax1, show=False)
	ax1.plot(tokamak.limiter.R, tokamak.limiter.Z, color='k', linewidth=1.2, linestyle="--")
	#ax1.plot(tokamak_alt.wall.R, tokamak_alt.wall.Z, color='k', linewidth=1.2, linestyle="-")

	ax1.grid(alpha=0.5)
	ax1.set_aspect('equal')
	ax1.set_xlim(0.0, 1.2)
	ax1.set_ylim(-1.5, 1.5)
	ax1.set_xlabel(r'Major radius, $R$ [m]')
	ax1.set_ylabel(r'Height, $Z$ [m]')


	plt.show()



def build_first_load(load_path):
	from freegsnke import build_machine
	load_path = Path(load_path)
	active_coils = load_path / 'active_coils.pickle'
	passive_coils = load_path / 'passive_coils.pickle'
	limiter = load_path / 'limiter.pickle'
	magnetic_probes = load_path / 'magnetic_probes.pickle'
	
	active_coils = 	 	str(active_coils)
	passive_coils = 	str(passive_coils)
	limiter = str(limiter)
	magnetic_probes = str(magnetic_probes)

	tokamak = build_machine.tokamak(
    active_coils_data=f"{active_coils}",
    passive_coils_data=passive_coils,
    limiter_data=limiter,
    #wall_data=wall,
    magnetic_probe_data=magnetic_probes,
	)
	return tokamak


def create_machine_files(save_path):
	active_coil =  active_coils()
	passive_coil = passive_coils()
	pickup_coils = bdots()
	flux_loops = flux_loop()
	lim = limiter()
	magnetic_probes = {'flux_loops': flux_loops, 'pickups': pickup_coils}
	save_machine(active_coil, passive_coil, lim, magnetic_probes,  save_path)


def build_first():
	active_coil =  active_coils_13_45()
	passive_coil = passive_coils()
	pickup_coils = bdots()
	flux_loops = flux_loop()
	lim = limiter(ellip=2.4)
	magnetic_probes = {'flux_loops': flux_loops, 'pickups': pickup_coils}


	for pc in pickup_coils:
		print(pc)
	from freegsnke import build_machine
	tokamak = build_machine.tokamak(
    active_coils_data=active_coil,
    passive_coils_data=passive_coil,
    limiter_data=lim,
    #wall_data=wall,
    magnetic_probe_data=magnetic_probes,
	)

	return tokamak

if __name__ == "__main__":
	#save_path = '/Users/hsiaohsl/workspace/pkg/freegsnke_v2.1.0/freegsnke/examples/first_machine'
	#create_machine_files(save_path)
	#tokamak_first = build_first_load(save_path)
	tokamak_first = build_first()
	plot_machine(tokamak_first)



	
