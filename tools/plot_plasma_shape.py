import numpy as np
from freeqdsk import geqdsk
import argparse
import matplotlib.pyplot as plt
import my_settings
cms = my_settings.cfg_my_settings()
my_colorbar = cms.my_colorbar
my_savefig = cms.my_savefig
from pathlib import Path
import glob
import os


def get_clevel(data_list):
	cmax = -np.inf
	cmin = np.inf
	for data in data_list:
		psi = data['psi']
		cmax = max(np.max(psi), cmax)
		cmin = min(np.min(psi), cmin)
	clevel = np.linspace(cmin, cmax, 50)
	return clevel

def plot_plasma_shape(ax, data, c_level):
	psi = data['psi']	
	r_grid = data['r_grid']
	z_grid = data['z_grid']
	psi_boundary = data['psi_boundary']
	rbdry = data['rbdry']	
	zbdry = data['zbdry']
	rbbbs = data['rbbbs']
	zbbbs = data['zbbbs']
	ax.set_aspect("equal")
	c_plot = ax.contourf(r_grid, z_grid, psi, c_level , cmap='plasma')
	ax.set_ylim(-1.2,1.2)
	ax.set_xlim(0,1.1)
	ax.plot(rbdry, zbdry)
	#ax.set_ylabel('Z[m]')
	ax.set_xlabel('R[m]')

def make_plot(root,savefile):
	file_list = glob.glob(os.path.join(root,'*geqdsk'))
	print(file_list)
	data_list = []
	for fn in file_list:
		f = open(fn, "r")
		data = geqdsk.read(f)
		data_list.append(data)
	clevel = get_clevel(data_list)
	
	num_figs = len(data_list)
	wide = num_figs*2.2
	height = 4
	fig, axs = plt.subplots(1, num_figs, figsize=(wide,height))
	for ax, data in zip(axs, data_list):
		plot_plasma_shape(ax, data, clevel)
	
	axs[0].set_ylabel('Z[m]')
	my_savefig(savefile)	


if __name__ == '__main__':
	root = '/Users/hsiaohsl/workspace/pkg/sana_narlabs/freegsnke_v2.1.0_nchc/examples/data_first/plot'
	#root = '/Users/hsiaohsl/workspace/pkg/sana_narlabs/tmp/freegsnke_v2.1.0_nchc/examples/data_first/plot'
	savefile = 'Ip5000_Btor0.1_am1an3.png'
	make_plot(root,savefile)
