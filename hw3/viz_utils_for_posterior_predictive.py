import numpy as np
import matplotlib.pyplot as plt

from neural_net_utils import predict_f_given_x

W = 6 # width of panel
H = 4 # height of panel

def plot_potential_energy_of_samples(history_PE_by_chain,
        sampler_options={
            'n_samples_keep':None,
            'n_samples_burnin':None},
        ):
    ''' Plot potential energy vs sample iteration, for diagnosing convergence

    Creates one line for each separate chain

    Args
    ----
    history_PE_by_chain : dict
        Keys are integers defining a chain number
        Values are lists of floats, representing potential energy values
        List contains entiry history of sampler

    Post Condition
    --------------
    Creates matplotlib plot in current notebook

    Returns
    -------
    None
    '''
    ncols = 2
    chain_list = list(history_PE_by_chain.keys())
    B = int(sampler_options['n_samples_burnin'])
    S = int(sampler_options['n_samples_keep'])
    _, axgrid = plt.subplots(
        nrows=1, ncols=2, figsize=(W*ncols+1, H), sharey=False)

    axL = axgrid[0]
    for chain in chain_list:
        axL.plot(history_PE_by_chain[chain], label='chain%02d' % chain)
    axL.set_xlabel('')
    axL.set_ylabel('potential energy');
    axL.legend(loc='upper right');

    axR = axgrid[1]
    for chain in chain_list:
        axR.plot(np.arange(B, B+S), history_PE_by_chain[chain][B:])
    axR.set_xlabel('iteration after burnin')
    axR.set_ylabel('potential energy');
    return axL, axR


def show_posterior_predictive_samples_and_intervals(
        history_nn_samples_by_chain, 
        y_ST_by_chain,
        x_grid_T=None,
        x_NF=None, y_N=None,
        sampler_options=None,
        ):
    ''' Plot potential energy vs sample iteration, for diagnosing convergence

    Creates one line for each separate chain

    Args
    ----
    history_nn_samples_by_chain : dict
        Keys are integers defining a chain number
        Values are lists of PyTrees, representing sampled NN weights/biases
        List contains entiry history of sampler

    Post Condition
    --------------
    Creates matplotlib plot in current notebook

    Returns
    -------
    None
    '''
    B = sampler_options['n_samples_burnin']
    S = sampler_options['n_samples_keep']

    # Show K=5 posterior samples
    # selected out of all samples at evenly spaced intervals
    # For example, if 200 kept samples, would select inds 0,50,100,150,199
    K = 5
    evenly_spaced_samples_K = np.minimum(np.arange(B,B+S+1,S//(K-1)), B+S-1)
    assert evenly_spaced_samples_K.size == K

    T = x_grid_T.size
    x_grid_T1 = x_grid_T.reshape((T,1))

    chain_list = list(history_nn_samples_by_chain.keys())
    
    # Create grid of axes
    nrows = len(chain_list)
    ncols = 2
    _, axgrid = plt.subplots(
        nrows=nrows, ncols=2, figsize=(W*ncols+1, H*nrows),
        squeeze=False, sharex=True, sharey=True)

    for cc, chain in enumerate(chain_list):

        # Left panel:
        # Show function implied by each of K=5 samples of the NN weights/biases
        # Will make 5 distinct lines, overlaid on the data
        axL = axgrid[cc,0]
        if cc == 0:
            axL.set_title(
                r"Predictions $f(x;\theta^s)$ via 5 posterior samples"
                + "\n" + r"where $\theta^s \sim p(\theta | y_{1:N}))$")
        axL.set_ylabel("chain %d" % chain)
        axL.plot(x_NF[:,0], y_N, 'ks')
        for ss in evenly_spaced_samples_K:
            nn_ss = history_nn_samples_by_chain[chain][ss]
            f_ss_T = predict_f_given_x(x_grid_T1, nn_ss)
            axL.plot(x_grid_T, f_ss_T, '-', linewidth=1)

        y_cc_ST = y_ST_by_chain[chain]
        # Compute mean and lo/hi of interval
        ymean_T = np.mean(y_cc_ST, axis=0)
        ylo_T = np.percentile(y_cc_ST, 2.5, axis=0)
        yhi_T = np.percentile(y_cc_ST, 97.5, axis=0)

        # Right panel:
        # Show mean and (2.5, 97.5) interval 
        axR = axgrid[cc,1]
        if cc == 0:
            axR.set_title(
                r"Posterior predictive $p(y_*|y_{1:N})$" 
                + "\n showing mean with 95% interval");
        axR.plot(x_NF[:,0], y_N, 'ks')
        axR.plot(x_grid_T, ymean_T, 'b-')
        axR.fill_between(x_grid_T, ylo_T, yhi_T, color='b', alpha=0.15)
