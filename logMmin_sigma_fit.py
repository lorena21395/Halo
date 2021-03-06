from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import numpy as np
from Corrfunc.theory.wp import wp
import scipy.optimize as op
import random
import zehavi_data_file_20
import warnings
from noisyopt import minimizeCompass,minimizeSPSA, bisect, AveragedFunction
warnings.filterwarnings("ignore")

wp_ng_vals = zehavi_data_file_20.get_wp()
bin_edges = zehavi_data_file_20.get_bins()
cov_matrix = zehavi_data_file_20.get_cov()
err = np.array([cov_matrix[i,i] for i in range(len(cov_matrix))])
bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.

#cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax')
cens_occ_model = Zheng07Cens()
cens_prof_model = TrivialPhaseSpace()

#sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'halo_vmax', modulate_with_cenocc=True)
sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True)
sats_prof_model = NFWPhaseSpace()

halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
halocat.redshift = 0.
pi_max = 60.
Lbox = 400.
model_instance = HodModelFactory(centrals_occupation = cens_occ_model, centrals_profile = cens_prof_model, satellites_occupation = sats_occ_model, satellites_profile = sats_prof_model)

try:
    model_instance.mock.populate()
except:
    model_instance.populate_mock(halocat)
        
alpha,logM0,logM1 = [1.16,13.28-1.7,13.28]
model_instance.param_dict['alpha'] = alpha
model_instance.param_dict['logM0'] = logM0
model_instance.param_dict['logM1'] = logM1

halo_table = halocat.halo_table
logMmin = np.logspace(11,14,100)

sig_fits = []
logMmin_list = []
func_val = []
def _find_sigma(guess):
    model_instance.param_dict['sigma_logM'] = guess
    model_instance.mock.populate()
    ng = model_instance.mock.number_density
    print("sigma: "+str(guess)[0:6]+"  density: " +str(ng)[0:7])
    sig_fits.append(guess)
    logMmin_list.append(m)
    ng_diff = wp_ng_vals[0]-ng
    func_val.append(ng_diff)
    return ng_diff

for m in logMmin:
    print("logMmin: "+ str(np.log10(m)))
    model_instance.param_dict['logMmin'] = np.log10(m)
    #res = minimizeCompass(_find_sigma,[0.3], bounds = [[0.01,1.5]],deltatol=0.0001, 
    #                      niter = 100, paired=False, disp=False, errorcontrol=False)
    #res = minimizeCompass(_find_sigma,[0.3], bounds = [[0.01,1.5]],
    #                      niter = 100, paired=False, disp=False, errorcontrol=False)
    avfunc = AveragedFunction(_find_sigma)
    res = bisect(avfunc, 0.01, 2., xtol=1e-06, errorcontrol=True,
                 outside='extrapolate', ascending=None, disp=False,
                 testkwargs={0.0001})

#res = op.minimize(_get_ng,[12.,0.30],options={'maxiter': 1e3,'disp':True},tol=1e-3)
#res = minimizeCompass(_get_ng,[12.36,0.38],bounds = [[11.0, 14.0], [0.01, 2.0]], deltatol=1e#-6, a=5.0, disp = True, paired=False)

np.save("logMmin_sigma_errctrl.npy",np.array([logMmin_list,sig_fits,func_val]))
