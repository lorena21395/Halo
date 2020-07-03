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
import zehavi_data_file_20
from numpy.linalg import inv

from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser(description='data type')
parser.add_argument('--file', type=str,
                    help='Name of data file to fit')
parser.add_argument('--catalog',type=str)
parser.add_argument('--output',type=str)
args = parser.parse_args()

import logging
log_fname = str(args.output[0:-2])+'log'
logger = logging.getLogger(str(args.output[0:-2]))
hdlr = logging.FileHandler(log_fname,mode='w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

if args.file == "zehavi":
    wp_ng_vals = zehavi_data_file_20.get_wp()
    bin_edges = zehavi_data_file_20.get_bins()
    cov_matrix = zehavi_data_file_20.get_cov()
    invcov = inv(cov_matrix)
    ng = wp_ng_vals[0]
    ng_cov = 0.00007
    wp_vals = wp_ng_vals[1:len(wp_ng_vals)]
elif args.file == "fake":
    wp_vals = MCMC_data_file.get_wp()
    bin_edges = MCMC_data_file.get_bins()
    cov_matrix = MCMC_data_file.get_cov()
    invcov = inv(cov_matrix)
    ng_cov = MCMC_data_file.get_ngcov()
    ng = MCMC_data_file.get_ng()
elif args.file == "guo":
    wp_ng_vals = guo_data_file.get_wp()
    bin_edges = guo_data_file.get_bins()
    cov_matrix = guo_data_file.get_cov()
    invcov = inv(cov_matrix)
    ng_cov = 0.00003
    wp_vals = wp_ng_vals[1:len(wp_ng_vals)]
    ng = wp_ng_vals[0]

cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax')
#cens_occ_model = Zheng07Cens()
cens_prof_model = TrivialPhaseSpace()

sats_occ_model =  Zheng07Sats(prim_haloprop_key = 'halo_vmax',modulate_with_cenocc=True)
#sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True)
sats_prof_model = NFWPhaseSpace()

model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                 centrals_profile = cens_prof_model,
                                 satellites_occupation = sats_occ_model,
                                 satellites_profile = sats_prof_model)

if args.catalog == "bolplanck":
    halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/bolplanck/rockstar/hlist_1.00231.list.halotools_v0p4.hdf5',update_cached_fname = True)
    halocat.redshift=0.
elif args.catalog == "old":
    halocat = CachedHaloCatalog(fname = '/home/lom31/Halo/hlist_1.00231.list.halotools_v0p1.hdf5',update_cached_fname = True)
    halocat.redshift=0.
elif args.catalog == "smdpl":
    halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
    halocat.redshift=0.
elif args.catalog  == "mdr1":
    halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/multidark/rockstar/hlist_0.68215.list.halotools_v0p4.hdf5',update_cached_fname = True)

try:
    model_instance.mock.populate()
except:
    model_instance.populate_mock(halocat)

def lnlike(theta):

    logMmin, sigma_logM, alpha, logM0, logM1 = theta
    model_instance.param_dict['logMmin'] = logMmin
    model_instance.param_dict['sigma_logM'] = sigma_logM
    model_instance.param_dict['alpha'] = alpha
    model_instance.param_dict['logM0'] = logM0
    model_instance.param_dict['logM1'] = logM1

    if args.catalog == 'smdpl':
        Lbox = 400.
    elif args.catalog == 'old':
        Lbox = 250.
    elif args.catalog == 'bolplanck':
        Lbox = 250.
    elif args.catalog == 'mdr1':
        Lbox=1000.
    model_instance.mock.populate()
    table = model_instance.mock.halo_table

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

    wp_calc = wp(Lbox,pi_max,nthreads,bin_edges,pos_zdist[:,0],pos_zdist[:,1],pos_zdist[:,2],verbose=False)#,xbin_refine_factor=3, ybin_refine_factor=3, zbin_refine_factor=2)

    wp_diff = wp_vals-wp_calc['wp']
    ng_diff = ng-model_instance.mock.number_density
    
    #save current wp calculated value for blobs
    global cur_wp
    cur_wp = wp_calc['wp'][:]

    return -0.5*np.dot(wp_diff, np.dot(invcov, wp_diff)) + -0.5*(ng_diff**2)/(ng_cov**2)

def lnprior(theta):
    logMmin, sigma_logM, alpha, logM0, logM1 = theta
    #if 1.0<logMmin<4.0 and 0.01<sigma_logM<2.5 and 1.0<alpha<5.0 and 1.0<logM0<4.0 and 1.0<logM1<4.0:
    if logMmin_r[0]<logMmin<logMmin_r[1] and sigma_logM_r[0]<sigma_logM<sigma_logM_r[1] and alpha_r[0]<alpha<alpha_r[1] and logM0_r[0]<logM0<logM0_r[1] and logM1_r[0]<logM1<logM1_r[1]:     
        return 0.0
    return -np.inf

def lnprob(theta):                      
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, cur_wp
    return lp + lnlike(theta),cur_wp

ndim, nwalkers, nsteps = 5, 20, 5e4
#####prior ranges
logMmin_r = [1.0,4.0]#[10.0,14.0]
sigma_logM_r = [0.01,2.5]
alpha_r = [0.85,5.0]
logM0_r = [1.0,4.0]
logM1_r = [1.0,4.0]

guess = 2.46, 1.38, 2.73, 1.30, 2.34
#guess =  11.83, 0.25, 1.0, 12.35, 13.08
pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


#write out configuration to log file
logger.info(args.output)
logger.info("guess: " + str(guess))
logger.info("ndim, nwalkers, nsteps: "+str(ndim)+","+str(nwalkers)+","+str(nsteps))
logger.info("logMmin_r: " + str(logMmin_r))
logger.info("sigma_logM_r: " + str(sigma_logM_r))
logger.info("alpha_r: " + str(alpha_r))
logger.info("logM0_r: " + str(logM0_r))
logger.info("logM1_r: " + str(logM1_r))


# Set up the backend
# Don't forget to clear it in case the file already exists
filename = args.output
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                    pool=pool,backend=backend)
    start = time.time()
    sampler.run_mcmc(pos, nsteps, progress=True, store=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

logger.info("Final size: {0}".format(backend.iteration))
