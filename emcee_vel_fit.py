from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import emcee
import corner
from Corrfunc.theory.wp import wp
import MCMC_data_file
from numpy.linalg import inv

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

wp_ng_vals = MCMC_data_file.get_wp()
bin_edges = MCMC_data_file.get_bins()[0:28]
cov_matrix = MCMC_data_file.get_cov()
invcov = inv(cov_matrix[1:28,1:28])
ng_cov = cov_matrix[0,0]

wp_vals = wp_ng_vals[1:28]
ng = wp_ng_vals[0]

cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax')
cens_prof_model = TrivialPhaseSpace()

sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'halo_vmax')
sats_prof_model = NFWPhaseSpace()

model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                 centrals_profile = cens_prof_model,
                                 satellites_occupation = sats_occ_model,
                                 satellites_profile = sats_prof_model)
halocat = CachedHaloCatalog(simname='bolplanck',redshift = 0.0)
model_instance.populate_mock(halocat)

def lnlike(theta):

    logMmin, sigma_logM, alpha, logM0, logM1 = theta
    model_instance.param_dict['logMmin'] = logMmin
    model_instance.param_dict['sigma_logM'] = sigma_logM
    model_instance.param_dict['alpha'] = alpha
    model_instance.param_dict['logM0'] = logM0
    model_instance.param_dict['logM1'] = logM1

    model_instance.mock.populate()
    table = model_instance.mock.halo_table
    pos = return_xyz_formatted_array(model_instance.mock.galaxy_table['x'],
                                     model_instance.mock.galaxy_table['y'],
                                     model_instance.mock.galaxy_table['z'])
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    velz = model_instance.mock.galaxy_table['vz']
    pos_zdist = return_xyz_formatted_array(x,y,z, velocity=velz,
                                velocity_distortion_dimension='z')
    pi_max = 60.
    Lbox = 250.
    nthreads = 1
    pos_zdist[:,2][np.where(pos_zdist[:,2]< 0.0)]= 0.0
    pos_zdist[:,2][np.where(pos_zdist[:,2]> 250.0)]= 250.0

    wp_calc = wp(Lbox,pi_max,nthreads,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2],verbose=False,xbin_refine_factor=3, ybin_refine_factor=3, zbin_refine_factor=2)

    f2 = open('current_params.txt','w')
    f2.write(str(theta))
    f2.close()

    wp_diff = wp_vals-wp_calc['wp']
    ng_diff = ng-model_instance.mock.number_density
    
    return -0.5*np.dot(wp_diff, np.dot(invcov, wp_diff)) + -0.5*(ng_diff**2)*ng_cov

def lnprior(theta):
    logMmin, sigma_logM, alpha, logM0, logM1 = theta
    if 1.<logMmin<3. and 0.2<sigma_logM<0.6 and 0.5<alpha<2.0 and 1.<logM0<3. and 1.0<logM1<4.:
        return 0.0
    return -np.inf

def lnprob(theta):                      
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

ndim, nwalkers = 5, 100
guess = [2.6, 0.4, 1.4, 2.3, 2.4]
pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "samples_2.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                    lnprob, pool=pool,backend=backend)
    start = time.time()
    sampler.run_mcmc(pos, 5000, progress=True, store=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    f2 = open('multiprocessing_time.txt','w')
    f2.write("Multiprocessing took {0:.1f} seconds".format(multi_time))
    f2.close()

samples = sampler.get_chain()
f = open('param.txt','w')
f.write(str(samples[np.where(sampler.lnprobability==sampler.lnprobability.max())]))
f.close()

print(samples[np.where(sampler.lnprobability==sampler.lnprobability.max())][0])
