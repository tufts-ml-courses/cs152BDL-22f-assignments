'''
Starter Code for an Autoencoder trained via reconstruction error (BCE)

TODO LIST
---------
* [ ] FIX calc_bce_loss
'''

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(
            self,
            n_dims_code=2,
            n_dims_data=784,
            hidden_layer_sizes=[32]):
        super(Autoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code

        self.kwargs = dict(
            n_dims_code=n_dims_code, n_dims_data=n_dims_data,
            hidden_layer_sizes=hidden_layer_sizes)

        encoder_layer_sizes = (
            [n_dims_data] + hidden_layer_sizes + [n_dims_code]
            )
        self.n_layers = len(encoder_layer_sizes) - 1

        # Create the encoder, layer by layer
        self.encoder_activations = list()
        self.encoder_params = nn.ModuleList()
        for layer_id, (n_in, n_out) in enumerate(zip(
                encoder_layer_sizes[:-1], encoder_layer_sizes[1:])):
            self.encoder_params.append(nn.Linear(n_in, n_out))
            self.encoder_activations.append(F.relu)
        self.encoder_activations[-1] = lambda a: a

        self.decoder_activations = list()
        self.decoder_params = nn.ModuleList()
        decoder_layer_sizes = [a for a in reversed(encoder_layer_sizes)]
        for (n_in, n_out) in zip(
                decoder_layer_sizes[:-1], decoder_layer_sizes[1:]):
            self.decoder_params.append(nn.Linear(n_in, n_out))
            self.decoder_activations.append(F.relu)
        self.decoder_activations[-1] = torch.sigmoid

    def forward(self, x):
        """ Run entire autoencoder on input (encode then decode)

        Returns
        -------
        x_vec : 1D array, size of x
        """
        return self.decode(self.encode(x))

    def encode(self, x_ND):
        cur_arr = x_ND
        for ll in range(self.n_layers):
            linear_func = self.encoder_params[ll]
            a_func = self.encoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        z_NC = cur_arr
        return z_NC

    def decode(self, z_NC):
        cur_arr = z_NC
        for ll in range(self.n_layers):
            linear_func = self.decoder_params[ll]
            a_func = self.decoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        xproba_ND = cur_arr
        return xproba_ND

    def calc_bce_loss(self, xbin_ND):
        ''' Compute binary cross entropy reconstruction loss for given data

        This is the signal we use to train the encoder/decoder.

        Returns
        -------
        bce : scalar
            Binary cross-entropy, summed over all images & pixels
        xproba_ND : tensor like xbin_ND, values within unit interval (0,1)
            Predicted probabilities after encoding then decoding input xbin_ND
        '''
        ## TODO write code to compute two things:
        # * the reconstruction 'xproba_ND' of the input binary array xbin_ND
        # * the binary cross entropy loss under this reconstruction
        xproba_ND = xbin_ND              # <-- TODO fix me
        
        bce = torch.sum(self.decoder_params[-1].weight)  # <-- TODO fix me
        return bce, xproba_ND

    def train_for_one_epoch_of_gradient_update_steps(
            self, optimizer, train_loader, device, epoch, args):
        ''' Perform one epoch of gradient updates on provided model & data.

        Steps through dataset, one minibatch at a time.
        At each minibatch, we compute the gradient and step in that direction.

        Post Condition
        --------------
        This object's internal parameters are updated
        '''
        self.train()
        train_loss = 0.0
        n_seen = 0
        num_batch_before_print = int(np.ceil(len(train_loader)/5))

        for batch_idx, (batch_data, _) in enumerate(train_loader):
            # Reshape the data from n_images x 28x28 to n_images x 784 (NxD)
            batch_x_ND = batch_data.to(device).view(-1, self.n_dims_data)

            # Zero out any stored gradients attached to the optimizer
            optimizer.zero_grad()

            # Compute the loss (and the required reconstruction as well)
            loss, batch_xproba_ND = self.calc_bce_loss(batch_x_ND)

            # Increment the total loss (over all batches)
            train_loss += loss.item()

            # Compute the gradient of the loss wrt model parameters
            # (gradients are stored as attributes of parameters of 'model')
            loss.backward()

            # Take an optimization step (gradient descent step)
            optimizer.step() # side-effect: updates internals of self's model!

            # Update total num images seen this epoch
            n_seen += batch_x_ND.shape[0]

            # Done with this batch. Write a progress update to stdout, move on.
            is_last = batch_idx + 1 == len(train_loader)
            if (batch_idx + 1) % num_batch_before_print  == 0 or is_last:
                l1_dist = torch.mean(torch.abs(batch_x_ND - batch_xproba_ND))
                print("  epoch %3d | frac_seen %.3f | avg loss %.3e | batch loss % .3e | batch l1 % .3f" % (
                    epoch, (1+batch_idx) / float(len(train_loader)),
                    train_loss / float(n_seen),
                    loss.item() / float(batch_x_ND.shape[0]),
                    l1_dist,
                    ))

    def save_to_file(self, fpath):
        """ Save this model to file
        """
        state_dict = self.state_dict()
        state_dict['kwargs'] = self.kwargs
        torch.save(state_dict, fpath)

    @classmethod
    def save_model_to_file(cls, model, fpath):
        """ Save given model to file (class method)
        """
        model.save_to_file(fpath)

    @classmethod
    def load_model_from_file(cls, fpath):
        """ Load from file (class method)

        Usage
        -----
        >>> Autoencoder.load_model_from_file('path/to/model.pytorch')
        """
        state_dict = torch.load(fpath)
        kwargs = state_dict.pop('kwargs')
        model = cls(**kwargs)
        assert 'kwargs' not in state_dict
        model.load_state_dict(state_dict)
        return model

    
