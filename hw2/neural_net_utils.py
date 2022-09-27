import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats


def predict_f_given_x(x_NF=None, nn_param_list=None, activation_func=jnp.tanh):
    """ Compute scalar output of a feed-forward neural net
    
    Args
    ----
    x_NF : 2D array, shape (N,F) = (n_examples, n_input_dims)
        F-dim feature vector for each example
    nn_param_list : list of dict
        Parameters of neural network
    activation_func : callable
        Activation function of the neural net
        Must be differentiable using jax.grad

    Returns
    -------
    f_N : 1D array, shape (N,) = (n_examples,)
        Predicted function value for each feature vector
    """
    assert x_NF.ndim == 2 # verify input features have shape (N,F)
    n_layers = len(nn_param_list)
    h_arr = x_NF
    i = 0
    for layer_dict in nn_param_list:
        # let i denote the current layer id
        i += 1
        # h_arr is the "input" to current layer, shape (N,J^i-1)
        # out_arr is "output" of the current layer, shape (N,J^i)
        # w is 2D array with shape (J^i, J^i-1)
        # b is 1D array with shape (J^i)
        out_arr = jnp.dot(h_arr, layer_dict['w']) + layer_dict['b']
        if i < n_layers:
            # If NOT the last layer, we apply activation func to each entry of out_arr
            h_arr = activation_func(out_arr)
    # Squeeze so returned shape becomes (N,) not (N,1)
    return jnp.squeeze(out_arr)

def make_nn_params_as_list_of_dicts(
        n_dims_input=1,
        n_dims_output=1,
        n_dims_per_hidden_list=[5],
        weight_fill_func=np.zeros,
        bias_fill_func=np.zeros):
    ''' Create a list of dicts structure defining a neural network
    
    Args
    ----
    n_dims_input : int
    n_dims_output : int
    n_hiddens_per_layer_list : list of int
    weight_fill_func : callable, like np.zeros
    bias_fill_func : callable, like np.zeros
    
    Returns
    -------
    nn_params : list of dicts
        Each dict has two keys, 'w' and 'b', for weights and biases
        The values are arrays of the specified shape.
    '''
    nn_param_list = []
    n_per_layer_list = [
        n_dims_input] + n_dims_per_hidden_list + [n_dims_output]
    
    # Given full network size list is [a, b, c, d, e]
    # Loop over adjacent pairs: (a,b) , (b,c) , (c,d) , (d,e)
    for n_in, n_out in zip(n_per_layer_list[:-1], n_per_layer_list[1:]):
        nn_param_list.append(
            dict(
                w=weight_fill_func((n_in, n_out)),
                b=bias_fill_func((n_out,)),
            ))
    return nn_param_list

def pretty_print_nn_param_list(nn_param_list_of_dict):
    ''' Create pretty display of the parameters at each layer
    '''
    with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
        for ll, layer_dict in enumerate(nn_param_list_of_dict):
            print("Layer %d" % ll)
            print("  w | shape %9s | %s" % (layer_dict['w'].shape, layer_dict['w'].flatten()))
            print("  b | shape %9s | %s" % (layer_dict['b'].shape, layer_dict['b'].flatten()))
            
def sample_nn_params_from_normal(
        n_dims_per_hidden_list,
        random_state=101,
        mean=0.0,
        stddev=1.0,
        ):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    
    def draw_from_Normal(shape):
        return random_state.normal(mean, stddev, size=shape)

    nn_params = make_nn_params_as_list_of_dicts(
        n_dims_input=1,
        n_dims_output=1,
        n_dims_per_hidden_list=n_dims_per_hidden_list,
        weight_fill_func=draw_from_Normal,
        bias_fill_func=draw_from_Normal)
    return nn_params
