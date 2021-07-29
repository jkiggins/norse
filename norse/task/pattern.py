import torch
from torch import nn

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPOptimizer

from norse.eval import logger, inspect

import pathlib
import argparse

def _lif_from_cfg(cfg):
    lif_params = LIFParameters(method=cfg['method'], v_reset=cfg['v_reset'], v_th=cfg['v_th'])
    return LIFCell(p=lif_params)
    

class PatternNet(torch.nn.Module):
    def __init__(self, cfg,
                 optimizer=None,
                 monitor=None,
                 dtype=torch.float):

        super(PatternNet, self).__init__()
        
        lif_cfg = cfg['lif']
        lif_params = LIFParameters(method=lif_cfg['method'], v_reset=lif_cfg['v_reset'])

        self.optimizer = optimizer
        self.monitor = monitor
        
        self._module_list = nn.ModuleList()
        self._lif_module_list = nn.ModuleList()
        self._lif_state_dict = {}

        # Create and init layers
        features = cfg['input_features']
        for l in cfg['layers']:
            layer = nn.Linear(features, l['features'], bias=l['bias'])
            lif_layer = _lif_from_cfg(cfg['lif'])

            if l['init'][0] == 'uniform':
                nn.init.uniform_(layer.weight)
            elif l['init'][0] == 'unity':
                print("init weights unity")
                layer.weight = nn.Parameter(torch.ones_like(layer.weight))
            elif l['init'][0] == 'normal':
                nn.init.normal_(layer.weight, mean=l['init'][1]['mu'], std=l['init'][1]['sigma'])
                
            torch.clip(layer.weight, 0.0, 1.0)

            self._module_list.append(layer)
            self._lif_module_list.append(lif_layer)

            self._lif_state_dict[lif_layer] = None
       
            features = l['features']


    def synapse_forward(self, pre, lif_state_pre, real_module, spiking_module, post, optimize=False, pre_is_input=False, name=None):
        if optimize and self.optimizer:
            self.optimizer(real_module, pre, post, name="fc{}".format(name))

        if self.monitor:
            self.monitor(pre, lif_state_pre, real_module, spiking_module, post, pre_is_input=pre_is_input, name=name)


    def forward(self, z, optimize=False):

        first_pass = True

        for i, (linear, lif) in enumerate(zip(self._module_list, self._lif_module_list)):
            lif_state_pre = self._lif_state_dict[lif]
            
            z_pre = z
            z = linear(z)
            z, lif_state = lif(z, lif_state_pre)

            self._lif_state_dict[lif] = lif_state
            
            self.synapse_forward(z_pre, lif_state_pre, linear, lif, z, optimize=optimize, pre_is_input=first_pass, name=str(i))
            first_pass = False

        return z


def _gen_spikes(pattern, features):
    def _get_constant(val, samples):
        return torch.ones(samples, features).type(torch.float)*1.0

    def _get_poisson(scale, samples):
        return torch.poisson(torch.rand((samples, features))*scale).type(torch.float)

    def _get_uniform_binary(th, samples):
        print("Uniform binary < {}, {} samples".format(th, samples))
        return (torch.rand(samples, features) < th)*1.0

    pattern_fn_dict = {
        'constant': _get_constant,
        'poisson': _get_poisson,
        'uniform': _get_uniform_binary
    }

    return pattern_fn_dict[pattern[0]](*pattern[1:])


def _spikes_from_cfg(cfg):
    spike_sets = []
    for pattern in cfg['input']['patterns']:
        spike_sets.append(_gen_spikes(pattern, cfg['input_features']))

    return spike_sets


def _load_config(path):
    path = pathlib.Path(path)

    if not path.exists():
        raise ValueError("Path: {} doesn't exist")


    with open(str(path), 'rb') as f:
        cfg = yaml.load(f)

    return cfg

    
def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--iters', type=int, default=100)
    
    return parser.parse_args()


def _main():
    args = _parse_args()
    
    spike_vector_len = 100
    num_iters = 100
    
    spike_monitor = logger.SpikeMonitor(weight_hist_iter=args.iters // 10)
    stdp_monitor = logger.STDPMonitor()
    stdp_optimizer = STDPOptimizer(alpha=1e-1, monitor=stdp_monitor)

    input_features = 20
    v_reset = min(-input_features / 8, -1.0)
    v_th = max(input_features / 8, 1.0)
    cfg = {
        'input': {
            'patterns': [('uniform', 0.5, 10), ('uniform', 0.2, 5)]
        },
        'input_features': input_features,
        'lif': {'method': 'super', 'v_reset': v_reset, 'v_th': v_th},
        'layers': [
            {'features': 1, 'bias': False, 'init': ('normal', {'mu': 0.5, 'sigma': 0.05})},
            # {'features': 2, 'bias': True, 'init': 'uniform'},
            # {'features': 1, 'bias': False, 'init': 'uniform'},
        ],
        'train': {'optimize': True, 'epochs': 100}
    }

    model = PatternNet(cfg, monitor=spike_monitor, optimizer=stdp_optimizer)

    target_patterns, luer = _spikes_from_cfg(cfg)
    spike_set = torch.vstack((target_patterns, luer))
    targets = torch.cat((torch.ones(target_patterns.shape[0:1]), torch.zeros(luer.shape[0:1])))

    # Train
    for epoch in range(cfg['train']['epochs']):
        # shuffle_idx = torch.randperm(spike_set.shape[0])
        # spike_set = spike_set[shuffle_idx, :]
        # targets = targets[shuffle_idx]
        
        for i, spike_vector in enumerate(spike_set):
            z = model(spike_vector, optimize=cfg['train']['optimize'])
            reward = ((z == targets[i])*1.0 - 0.5)*2.0
            print(reward)
            stdp_optimizer.step_reward(reward)

    # Clear out model state
    for i in range(100):
        model(torch.zeros(cfg['input_features']), optimize=False)

    # Test trained patterns
    print("Pattern spike outputs: ")
    for i, spike_vector in enumerate(spike_set):
        for j in range(2):
            z = model(spike_vector, optimize=False)
            
        print(z)

        # Clear out model state
        for i in range(100):
            model(torch.zeros(cfg['input_features']), optimize=False)

    
        

if __name__ == '__main__':
    path = './runs/pattern'
    
    logger.set_path(path)
    
    with torch.no_grad():
        _main()
