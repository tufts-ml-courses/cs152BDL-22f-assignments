import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from neural_net_utils import predict_f_given_x, make_nn_params_as_list_of_dicts

def sample_nn_params_from_prior(
        n_dims_per_hidden_list,
        random_state=101,
        prior_params={'w_mean':0.0, 'w_stddev':1.0,
                      'b_mean':0.0, 'b_stddev':1.0},
        ):
    w_mean, w_stddev = prior_params['w_mean'], prior_params['w_stddev']
    b_mean, b_stddev = prior_params['b_mean'], prior_params['b_stddev']
    def draw_weights_from_Normal_prior(shape, persistent_state={'seed':int(random_state)}):
        # increment the seed
        persistent_state['seed'] += 1
        prng = np.random.RandomState(persistent_state['seed'])
        return prng.normal(w_mean, w_stddev, size=shape)
    def draw_biases_from_Normal_prior(shape, persistent_state={'seed':int(random_state)+1001}):
        # increment the seed
        persistent_state['seed'] += 1
        prng = np.random.RandomState(persistent_state['seed'])
        return prng.normal(b_mean, b_stddev, size=shape)

    nn_params = make_nn_params_as_list_of_dicts(
        n_dims_input=1,
        n_dims_output=1,
        n_dims_per_hidden_list=n_dims_per_hidden_list,
        weight_fill_func=draw_weights_from_Normal_prior,
        bias_fill_func=draw_biases_from_Normal_prior)
    return nn_params
        
def calc_logpdf_likelihood(
        nn_params, x_NF, y_N,
        lik_params={'tau':.1}):
    ''' Compute scalar logpdf of provided outputs given weights/biases
    
    Returns
    -------
    logpdf : scalar float
       log probability density function of likelihood
    '''
    f_N = predict_f_given_x(x_NF, nn_params)
    logpdf_N = jstats.norm.logpdf(y_N, f_N, lik_params['tau'])
    return jnp.sum(logpdf_N)

def calc_logpdf_prior(nn_params,
        prior_params={
            'w_mean':0.0, 'w_stddev':1.0,
            'b_mean':0.0, 'b_stddev':1.0}):
    ''' Compute scalar logpdf of provided weights/biases under prior
    
    Returns
    -------
    logpdf : scalar float
    '''
    L = len(nn_params)
    total_logpdf = 0.0
    for ll in range(L):
        w_arr = nn_params[ll]['w']
        b_arr = nn_params[ll]['b']
        total_logpdf += jnp.sum(jstats.norm.logpdf(
            w_arr, prior_params['w_mean'], prior_params['w_stddev']))
        total_logpdf += jnp.sum(jstats.norm.logpdf(
            b_arr, prior_params['b_mean'], prior_params['b_stddev']))
    return total_logpdf