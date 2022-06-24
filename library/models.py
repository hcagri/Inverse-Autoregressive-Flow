import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal


class MaskedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def set_mask(self, mask):
        self.mask = mask

    def shape(self):
        return self.linear.weight.size()

    def forward(self, x):
        self.linear.weight = Parameter(data=(self.linear.weight * self.mask))
        return self.linear(x)


class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, context_h_dim = None, natural_ordering = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.natural_ordering = natural_ordering

        if context_h_dim:
            self.h_net = nn.Linear(context_h_dim, hidden_dims[0])
        
        self.masked_layers = nn.ModuleList()

        for h_dim in self.hidden_dims:
            self.masked_layers.append(MaskedLayer(input_dim, h_dim))
            input_dim = h_dim
        self.masked_layers.append(MaskedLayer(h_dim, output_dim))

        self.dropout = nn.Dropout(0.2)

        self.assign_masks()

    def assign_masks(self):

        indexes = [torch.arange(self.input_dim) if self.natural_ordering else torch.randperm(self.input_dim)]
        for h_dim in self.hidden_dims:
            indexes.append(torch.randint(indexes[-1].min(), self.input_dim, size=(h_dim,)))

        # Since we output 2 different values [sigma, mu] each of them are created by autoregressive network
        indexes.append(torch.cat([indexes[0], indexes[0]]))


        for i in range(len(self.masked_layers)):

            mask = torch.empty(size=self.masked_layers[i].shape())

            index_prev, index_next = indexes[i:i+2]

            for j, index_value in enumerate(index_prev):
                
                if i != len(self.masked_layers) - 1:
                    mask[:, j] = (index_next >= index_value).clone().detach()
         
                else:
                    # Last layer, strictly smaller
                    mask[:, j] = (index_next > index_value).clone().detach()
            
            self.masked_layers[i].set_mask(mask)
        

    def forward(self, x, h=None):
        
        out = self.masked_layers[0](x)
        if h is not None:
            h_out = self.h_net(h)
            out = out + h_out
        out = F.elu(out)

        for i in range(1,len(self.masked_layers) - 1):
            out = F.elu(self.masked_layers[i](out))
            # out = self.dropout(out)
    
        out = self.masked_layers[-1](out)
        return out


class IAFLayer(nn.Module):
    """ A class implementing a single layer of Inverse Autoregressive Flow """

    def __init__(self, encoding_dim, hidden_dims, context_h_dim=None):
        """
        Initialize the IAF layer with any architecture you want. This class will 
        have two operations: up() and down(). You need separate NNs for up() and
        down().
        """
        super().__init__()
        self.e_dim = encoding_dim
        # I'm not putting super().__init__() here anymore!
        self.made = MADE(input_dim=encoding_dim, hidden_dims=hidden_dims, output_dim=2*encoding_dim, context_h_dim=context_h_dim)


    def up(self, z, h):
        """ Given h and z, calculate new z and any other necessary outputs """

        ms = self.made(z, h)
        m,s = ms.chunk(2, dim=1)
        sigma = s.exp()
        z_t = sigma*z + (1-sigma)*m 

        log_sigma = torch.sum(torch.log(sigma), dim=1)

        return z_t, log_sigma

    def down(self, h, z):
        """ Given h and z, calculate new z and any other necessary outputs """
        pass


class VAE(nn.Module):
    """ 
    A variational autoencoder that uses IAF layers.
    """

    def __init__(self, params):
        """ 
        Initialize first (encoder) and last (decoder) layers, together with
        the IAF layers.
        """
        super().__init__()

        in_dim = params["input_dim"]

        self.encoder = []
        for i, h_dim in enumerate(params['encoder']):
            if i != len(params['encoder']) - 1:
                self.encoder.append(nn.Linear(in_dim, h_dim))
                self.encoder.append(nn.ELU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*self.encoder)

        self.gaus_head = nn.Linear(params['encoder'][-2], 2*h_dim)
        self.context_head = nn.Linear(params['encoder'][-2], params['context_h_dim'])
        
        self.flows  = nn.ModuleList()

        for _ in range(params["num_flows"]):
            self.flows.append(IAFLayer(h_dim, params["iaf_hidden_dims"], params['context_h_dim']))


        self.decoder = []
        for i, h_dim in enumerate(params['decoder']):
            
            self.decoder.append(nn.Linear(in_dim, h_dim))
            if i != len(params['decoder']) - 1:
                self.decoder.append(nn.ELU())
            in_dim = h_dim
        self.decoder = nn.Sequential(*self.decoder)

    
    def reparametrize(self, mu, log_var):

        eps = MultivariateNormal(torch.zeros_like(mu), torch.eye(mu.size()[1])).sample()
        std = torch.exp(log_var)
        z = eps*std + mu 

        return eps, z

    def forward(self, x):
        """
        A full forward pass of the VAE. Compute ELBO.
        """
        # Encoder outputs log of variance to make sure that we have a positive variance
        out_enc = self.encoder(x)

        h = F.elu(self.context_head(out_enc))
        mu, log_var = F.elu(self.gaus_head(out_enc)).chunk(2,dim=1)

        eps, z = self.reparametrize(mu, log_var)

        # Taking logarithm of exponential of log_var = log_var if log_var > 0, however we make sure that we get a positive log_var value if log_var < 0
        l = -torch.sum(torch.log(torch.exp(log_var)) + 0.5*eps**2 + 0.5*torch.log(2*torch.tensor(torch.pi)), dim=1)

        for flow in self.flows:
            z, log_los = flow.up(z, h)
            l -= log_los
        
        log_p_zt = -torch.sum(0.5*z**2 + 0.5*torch.log(2*torch.tensor(torch.pi)), axis = 1)
        out = self.decoder(z)
        return out, log_p_zt, l

