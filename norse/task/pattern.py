import torch
from torch import nn

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPOptimizer

from matplotlib import pyplot as plt

from norse.eval import logger

import pathlib
import argparse

def _lif_from_cfg(cfg, dt=0.001):
    lif_params = LIFParameters(method=cfg['method'], v_reset=cfg['v_reset'], v_th=cfg['v_th'])
    return LIFCell(p=lif_params, dt=dt)
    

class PatternNet(torch.nn.Module):
    def __init__(self, cfg,
                 optimizer=None,
                 dtype=torch.float,
                 dt=0.001):

        super(PatternNet, self).__init__()
        
        lif_cfg = cfg['lif']

        self.optimizer = optimizer
        self.monitors = []
        
        self._module_list = nn.ModuleList()
        self._lif_module_list = nn.ModuleList()
        self._lif_state_dict = {}

        # Create and init layers
        features = cfg['input_features']
        for l in cfg['layers']:
            layer = nn.Linear(features, l['features'], bias=l['bias'])
            lif_layer = _lif_from_cfg(cfg['lif'], dt=dt)

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


    def add_monitor(self, monitor):
        self.monitors.append(monitor)

    def stop_monitoring(self):
        for m in self.monitors:
            m.stop()

            
    def reset_state(self):
        print("Reset LIF State")
        for key in self._lif_state_dict:
            self._lif_state_dict[key] = None
            

    def resume_monitoring(self):
        for m in self.monitors:
            m.resume()


    def synapse_forward(self, pre, lif_state_pre, real_module, spiking_module, post, optimize=False, pre_is_input=False, name=None):
        if optimize and self.optimizer:
            self.optimizer(real_module, pre, post, name="fc{}".format(name))

        for monitor in self.monitors:
            monitor(pre, lif_state_pre, real_module, spiking_module, post, pre_is_input=pre_is_input, name=name)


    def forward(self, z, optimize=False):

        first_pass = True

        for i, (linear, lif) in enumerate(zip(self._module_list, self._lif_module_list)):
            lif_state_pre = self._lif_state_dict[lif]
            
            z_pre = z
            z = linear(z)
            z, lif_state = lif(z, lif_state_pre)

            self._lif_state_dict[lif] = lif_state
            
            self.synapse_forward(z_pre, lif_state, linear, lif, z, optimize=optimize, pre_is_input=first_pass, name=str(i))
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

    def _get_onehot(index):
        spikes = torch.zeros(features) * False
        spikes[index] = True

        return spikes.view(1, -1)
    
        

    pattern_fn_dict = {
        'constant': _get_constant,
        'poisson': _get_poisson,
        'uniform': _get_uniform_binary,
        'onehot': _get_onehot,
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


def _eval(patterns, targets, model, cfg, repeat=10):
    correct = []
    model.stop_monitoring()
    for i, pat in enumerate(patterns):
        model.reset_state()
        for j in range(repeat):
            model(pat)
            z = model(pat)
            print("z: {}, target: {}".format(z, targets[i]))
            correct.append(z == targets[i])

    correct = torch.vstack(correct)*1.0
    print("accuracy: ", correct.mean())
    model.resume_monitoring()


def _pack_spike_samples(spike_samples, padding=10):
    for sample in spike_samples:
        for i in range(padding):
            yield torch.zeros_like(sample)

        yield sample

    
def _train(model, cfg, inputs, targets, optim, repeat=10):

    print("\nTraining")
    # Train
    for epoch in range(cfg['train']['epochs']):
        # shuffle_idx = torch.randperm(spike_set.shape[0])
        # spike_set = spike_set[shuffle_idx, :]
        # targets = targets[shuffle_idx]

        for i, spike_vector in enumerate(inputs):
            model.reset_state()
            optim.reset_state()
            for j in range(repeat):
                z = model(spike_vector, optimize=cfg['train']['optimize'])
                z = model(torch.zeros_like(spike_vector), optimize=cfg['train']['optimize'])
                reward = ((z == targets[i])*1.0 - 0.5)*2.0
                optim.step(reward=reward)

            # fig = plt.Figure()
            # ax = spike_monitor.graph_timeline(ax=fig.add_subplot())
            # logger.savefig(fig, "epoch_{}_spikes.png".format(epoch))

    print("\nAfter Training")
    model.reset_state()
    _eval(inputs, targets, model, cfg, repeat=repeat)
    
    
def _main():
    args = _parse_args()
    
    spike_vector_len = 100
    num_iters = 100
    
    spike_monitor = logger.SpikeMonitor(weight_hist_iter=args.iters // 10)

    input_features = 2
    v_reset = -0.3
    v_th = 0.3
    
    cfg = {
        'input': {
            'patterns': [('onehot', 0), ('onehot', 1)]
        },
        'input_features': input_features,
        'lif': {'method': 'super', 'v_reset': v_reset, 'v_th': v_th},
        'layers': [
            {'features': 1, 'bias': False, 'init': ('normal', {'mu': 0.5, 'sigma': 0.05})},
            # {'features': 2, 'bias': True, 'init': 'uniform'},
            # {'features': 1, 'bias': False, 'init': 'uniform'},
        ],
        'train': {'optimize': True, 'epochs': 5}
    }

    target_patterns, luer = _spikes_from_cfg(cfg)
    
    spike_set = torch.vstack((target_patterns, luer))
    targets = torch.cat((torch.ones(target_patterns.shape[0:1]), torch.zeros(luer.shape[0:1])))

    stdp_monitor = logger.STDPMonitor()
    stdp_optimizer = STDPOptimizer(alpha=1e-1, decay_fn='recent', monitor=stdp_monitor)

    model = PatternNet(cfg, dt=0.0001, optimizer=stdp_optimizer)
    model.add_monitor(spike_monitor)
    
    _train(model, cfg, spike_set, targets, stdp_optimizer, repeat=1)
        

if __name__ == '__main__':
    path = './runs/pattern'
    
    logger.set_path(path)
    
    with torch.no_grad():
        _main()