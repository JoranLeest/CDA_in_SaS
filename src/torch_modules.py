import torch
import numpy as np
import random
import torch.nn as nn
from collections.abc import Iterable


class MLP(nn.Module):
    '''
    This code was adapted from the code used in:
        
    Hasan Tercan, Philipp Deibert, and Tobias Meisen. 2022. Continual learning
    of neural networks for quality prediction in production using memory aware
    synapses and weight transfer. Journal of Intelligent Manufacturing 33, 1 (2022),
    283–292.
    '''
    
    def __init__(self, input_dim, hidden_dims, output_dim, n_heads=1):
        super().__init__()

        self.n_hidden = len(hidden_dims)
        self.n_heads = n_heads

        self.active_head = 0

        self.layer_dims = list(zip( [input_dim] + hidden_dims,
                                    hidden_dims + [output_dim] ))

        self.layers = nn.ModuleDict()
        
        for i, dims in enumerate(self.layer_dims[:-1], 1):
            self.layers.add_module('h{}'.format(i), nn.Linear(*dims))
        
        for i, dims in enumerate([self.layer_dims[-1]]*n_heads, 1):
            self.layers.add_module('out{}'.format(i), nn.Linear(*dims))

        self.activation = nn.ReLU()

    def forward(self, x, head=None):
        for i in range(self.n_hidden):
            x = self.layers[ 'h{}'.format(i+1) ](x)
            x = self.activation(x)
        if(head is None):
            x = self.layers[ 'out{}'.format(self.active_head + 1) ](x)
        elif(isinstance(head, int)):
            x = self.layers[ 'out{}'.format(head+1) ](x)
        elif(isinstance(head, str) and head == 'all'):
            head = list(range(self.n_heads))
        elif(isinstance(head, Iterable)):
            x = tuple(self.layers[ 'out{}'.format(h+1) ](x) for h in head)
        else:
            raise TypeError("Unknown type '{}' of argument 'head'".format(type(head)))

        return x

    def add_head(self, n=1):
        dims = self.layer_dims[-1]

        for i in range(0, n):
            self.n_heads += 1
            self.layers.add_module('out{}'.format(self.n_heads), nn.Linear(*dims))

    def use_head(self, i):
        self.active_head = i

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def store(self, path):
        torch.save(self.state_dict(), path)
        

class MAS(MLP):
    '''
    This code was adapted from the code used in:
        
    Hasan Tercan, Philipp Deibert, and Tobias Meisen. 2022. Continual learning
    of neural networks for quality prediction in production using memory aware
    synapses and weight transfer. Journal of Intelligent Manufacturing 33, 1 (2022),
    283–292.
    '''
    def __init__(self, input_dim, hidden_dims, output_dim, n_heads=1):
        super().__init__(input_dim, hidden_dims, output_dim, n_heads)
        self.init_omega_and_theta()
        
    def init_omega_and_theta(self):
        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}

        for name, param in self.named_parameters():
            if('omega_{}'.format(name.replace('.', '-')) not in omega_dict):
                self.register_buffer( 'omega_{}'.format(name.replace('.', '-')), torch.zeros_like(param, requires_grad=False) )
            if('theta_{}'.format(name.replace('.', '-')) not in theta_dict):
                self.register_buffer( 'theta_{}'.format(name.replace('.', '-')), param.clone().detach() )

    def update_theta(self):
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}

        for name, param in self.named_parameters():
            theta = theta_dict['theta_{}'.format(name.replace('.', '-'))]
            theta.data = param.clone().detach()

    def update_omega(self, data_loader: torch.utils.data.DataLoader, task_id_dict=None, gamma=1.0, use_task_id=True, accumulate=True, device=None):
        criterion = torch.nn.MSELoss(reduction='sum')

        for name, param in self.named_parameters():
            param.grad = None

        mode = self.training
        self.eval()

        n_samples = 0

        for i, (t, x, _) in enumerate(data_loader):
            if(task_id_dict is not None):
                t = tuple(task_id_dict[int(sample)] for sample in t)

            tasks = sorted(set(t))
            
            n_samples += x.shape[0]

            x = x.to(device)

            if(use_task_id):
                out = self(x.float(), head=tasks)

                out = torch.stack([out[tasks.index(t_id)][i] for i, t_id in enumerate(t)], dim=0) if len(out) > 1 else out[0]
            else:
                out = self(x, head='all')
                out = torch.cat(out, axis=1)

            zeros = torch.zeros(out.size()).to(device)
            loss = criterion(out, zeros)
            loss.backward()

        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}

        for name, param in self.named_parameters():

            omega = omega_dict['omega_{}'.format(name.replace('.', '-'))]

            if(param.grad is not None):
                if(accumulate):
                    omega.data *= gamma
                    omega.data += torch.abs(param.grad.detach()) / n_samples
                else:
                    omega.data = torch.abs(param.grad.detach()) / n_samples

            param.grad = None

        self.train(mode)

    def compute_omega_loss(self):
        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}

        omega_loss = 0.0

        for name, param in self.named_parameters():
            omega = omega_dict['omega_{}'.format(name.replace('.', '-'))]
            theta = theta_dict['theta_{}'.format(name.replace('.', '-'))]

            omega_loss += torch.sum( ((param-theta)**2) * omega )

        return omega_loss
    
    def add_head(self, n=1):
        super().add_head(n)
        self.init_omega_and_theta()
    
    