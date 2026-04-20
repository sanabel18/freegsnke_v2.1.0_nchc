import matplotlib.pyplot as plt
import numpy as np
import os
from freeqdsk import geqdsk
import my_settings
cms = my_settings.cfg_my_settings()
my_colorbar = cms.my_colorbar
my_savefig = cms.my_savefig
from pathlib import Path
import matplotlib.transforms as mtrans
from matplotlib.text import TextPath

def read_data_list(file_list):
	#print('file_list {}'.format(file_list))
	data_list = []
	label_list = []
	for fp in file_list:
		fo = open(fp, 'r')	
		data = geqdsk.read(fo)
		name = Path(fp).name
		name = name.replace('#','')
		labels = name.split('_')
		#print(labels[-1])
		label = '{}'.format('\#{}'.format(labels[-1]))
		data_list.append(data)
		label_list.append(label)
	return data_list, label_list


def first_wall():
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

	R_wall = np.concatenate((R_pos_inner, R_pos_arc), axis=0)	
	Z_wall = np.concatenate((Z_pos_inner, Z_pos_arc), axis=0)	
	return R_wall, Z_wall


def plot(ax, data):
	psi = data['psi']
	rbdry = data['rbdry']
	zbdry = data['zbdry']
	r_grid = data['r_grid']
	z_grid = data['z_grid']
	rp_idx = np.where(r_grid[:,0]<1.5)
	zp_idx = np.where(z_grid[0,:] < 2)
	zp_idx = np.where(zp_idx[0] > -2)
	print('rp_idx {}'.format(rp_idx))
	print('zp_idx {}'.format(zp_idx))

	Rwall, Zwall = first_wall()	
	zmin = 75
	zmax = 182
	rmax = 140
	ax.pcolormesh(r_grid[0:rmax,zmin:zmax], z_grid[0:rmax,zmin:zmax], psi[0:rmax,zmin:zmax], cmap='plasma')	
	levels = np.linspace(np.min(psi), np.max(psi), 50)
	#ct = ax.contour(r_grid[0:150,50:200], z_grid[0:150,50:200], psi[0:150,50:200], color = 'r', levels=levels)
	ax.plot(rbdry, zbdry)
	#ax.scatter(Rwall, Zwall, s=.3, color='black')
	ax.set_aspect('equal')
	ax.set_xlabel('R [m]')
	ax.set_ylabel('Z [m]')
	plt.tight_layout()		
	#plt.show()


if __name__ == "__main__":
	import glob
	root = '/Users/hsiaohsl/workspace/pkg/sana_narlabs/freegsnke_v2.1.0_nchc/examples/data'
	root = '/Users/hsiaohsl/workspace/pkg/sana_narlabs/tmp/freegsnke_v2.1.0_nchc/examples/data_first/plot'
	file_list = glob.glob(os.path.join(root,'*geqdsk'))
	print(file_list)
	data_list, lbl_list = read_data_list(file_list)
	fig, axs = plt.subplots(1,3, figsize=(8,6))
	for data, ax in zip(data_list, axs):
		plot(ax, data)
	plt.show()
