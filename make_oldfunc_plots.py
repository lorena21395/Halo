from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import numpy as np
import matplotlib.pyplot as plt
#from Corrfunc._countpairs import countpairs as wp
from Corrfunc.theory import wp
from numpy.linalg import inv
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#matplotlib.use('Agg')

import zehavi_data_file_20

wp_ng_vals = zehavi_data_file_20.get_wp()
bin_edges = zehavi_data_file_20.get_bins()
cov_matrix = zehavi_data_file_20.get_cov()
invcov = inv(cov_matrix)
ng = wp_ng_vals[0]
ng_cov = 0.00007
wp_vals = wp_ng_vals[1:len(wp_ng_vals)]

#halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
halocat = CachedHaloCatalog(fname='/home/lom31/.astropy/cache/halotools/halo_catalogs/bolplanck/rockstar/hlist_1.00231.list.halotools_v0p4.hdf5')
halocat.redshift=0.

cens_occ_model = Zheng07Cens()
cens_prof_model = TrivialPhaseSpace()
sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True)
sats_prof_model = NFWPhaseSpace()

model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                 centrals_profile = cens_prof_model,
                                 satellites_occupation = sats_occ_model,
                                 satellites_profile = sats_prof_model)
try:
    model_instance.mock.populate()
except:
    model_instance.populate_mock(halocat)

theta = [11.96,0.38,1.16,13.28-1.7,13.28]
logMmin, sigma_logM, alpha, logM0, logM1 = theta
model_instance.param_dict['logMmin'] = logMmin
model_instance.param_dict['sigma_logM'] = sigma_logM
model_instance.param_dict['alpha'] = alpha
model_instance.param_dict['logM0'] = logM0
model_instance.param_dict['logM1'] = logM1

Lbox = 250.
model_instance.mock.populate()
pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'],
                                 model_instance.mock.galaxy_table['y'],
                                 model_instance.mock.galaxy_table['z'],
                                 period = Lbox)
x = pos[:,0]
y = pos[:,1]
z = pos[:,2]
velz = model_instance.mock.galaxy_table['vz']
pos_zdist = return_xyz_formatted_array(x,y,z,period=Lbox,velocity=velz,
                                       velocity_distortion_dimension='z')
pi_max = 60.
nthreads = 4
import halotools,Corrfunc
print(Corrfunc.__version__)
wp_calc = wp(Lbox,pi_max,nthreads,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2])
#bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.

#plt.plot(bin_cen,wp_calc['wp'])
#plt.errorbar(bin_cen,wp_ng_vals[1:len(wp_ng_vals)],yerr=np.sqrt(err),fmt='o',markersize=2,capsize=4,label='data')
#plt.savefig('oldfunc_20.png')
print(wp_calc['wp'])
import halotools,Corrfunc
print(halotools.__version__)
print(Corrfunc.__version__)
