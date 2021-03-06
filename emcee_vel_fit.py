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

wp_ng_vals = MCMC_data_file.get_wp()
bin_edges = MCMC_data_file.get_bins()
cov_matrix = MCMC_data_file.get_cov()

bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.
err = np.sqrt(np.array([cov_matrix[i,i] for i in range(len(cov_matrix))]))
wp_vals = wp_ng_vals[1:len(wp_ng_vals)]
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

def lnlike(theta):#, wp_val, wperr, model_instance):                                 
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
    wp_calc = wp(Lbox,pi_max,nthreads,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2])
    inv_sigma2 = 1.0/err**2.

    wp_val = np.zeros(30)
    wp_val[0] = model_instance.mock.number_density
    wp_val[1:30] =+ wp_calc['wp']

    return -0.5*(np.sum((wp_ng_vals-wp_val**2*inv_sigma2 - np.log(inv_sigma2))))

def lnprior(theta):
    logMmin, sigma_logM, alpha, logM0, logM1 = theta
    if 1.<logMmin<3. and 0.2<sigma_logM<0.6 and 0.8<alpha<1.5 and 1.<logM0<3. and 2.5<logM1<4.:
        return 0.0
    return -np.inf

def lnprob(theta):                      
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

ndim, nwalkers = 5, 100
guess = [2.1,0.38,1.16,1.8,3.2]
pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "samples.h5"
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
