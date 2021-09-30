
# Torch imports
import torch
from torch import nn

# Norse Imports
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPOptimizer
from norse.eval import logger
from norse.torch.utils import registry
from norse.torch.module.astrocyte import Astrocyte

# Other Libs
from matplotlib import pyplot as plt

from norse.eval import logger

import pathlib
import argparse
import oyaml as yaml
from pprint import pprint


def _lif_from_params(cfg, dt=0.001):
    lif_params = LIFParameters(method=cfg['method'], v_reset=cfg['v_reset'], v_th=cfg['v_th'])
    
    return LIFCell(p=lif_params, dt=dt)


def _linear_from_params(in_features, out_features, params):
    layer = nn.Linear(in_features, out_features, bias=params['bias'])
    
    if params['init'][0] == 'uniform':
        nn.init.uniform_(layer.weight)
    elif params['init'][0] == 'unity':
        print("init weights unity")
        layer.weight = nn.Parameter(torch.ones_like(layer.weight))
    elif params['init'][0] == 'normal':
        mu = params['init'][1]
        sigma = params['init'][2]
        nn.init.normal_(layer.weight, mean=mu, std=sigma)
                    
    torch.clip(layer.weight, 0.0, 1.0)

    return layer


class PatternNet(torch.nn.Module):
    # TODO: Convert to from_cfg format
    def __init__(self, cfg,
                 optimizer=None,
                 dtype=torch.float):

        super(PatternNet, self).__init__()

        dt = cfg['sim']['dt']
        
        self.optimizer = optimizer
        self.monitors = []
        
        self._module_list = nn.ModuleList()
        self._lif_module_list = nn.ModuleList()
        self._lif_state_dict = {}

        self.snn_modules = []

        # Create and init layers
        for key, l in cfg['topology'].items():
            # Create layers
            linear_layer = None
            lif_layer = None
        
            if key == 'input':
                if l['type'] == 'lif':
                    lif_layer = _lif_from_params(cfg['lif_params'], dt=dt)
                    
            elif l['type'] == 'lif':
                linear_layer = _linear_from_params(features, l['num'], cfg['linear_params'])
                lif_layer = _lif_from_params(cfg['lif_params'], dt=dt)

            # Fill various lists
            if linear_layer:
                self._module_list.append(linear_layer)

            if lif_layer:
                self._lif_module_list.append(lif_layer)
                self._lif_state_dict[lif_layer] = None

            self.snn_modules.append((linear_layer, lif_layer))
       
            features = l['num']


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


def _gen_spikes(*pattern):
    def _get_constant(val, samples, features):
        return torch.ones(samples, features).type(torch.float)*1.0

    def _get_poisson(scale, samples, features):
        return torch.poisson(torch.rand((samples, features))*scale).type(torch.float)

    def _get_uniform_binary(th, samples, features):
        print("Uniform binary < {}, {} samples".format(th, samples))
        return (torch.rand(samples, features) < th)*1.0

    def _get_onehot(features):
        spikes = torch.eye(features)

        return spikes

    pattern_fn_dict = {
        'constant': _get_constant,
        'poisson': _get_poisson,
        'uniform': _get_uniform_binary,
        'onehot': _get_onehot,
    }

    arg_list = ()
    if len(pattern) >= 2:
        arg_list = pattern[1:]

    return pattern_fn_dict[pattern[0]](*arg_list)


def _sim_spikes(features):
    def generator():
        spike_idx = 0
        spike_delta = 1

        while True:
            spikes = torch.zeros(features)
            spikes[spike_idx] = 1

            if (spike_idx+spike_delta) >= features or (spike_idx+spike_delta) < 0:
                spike_delta = -spike_delta
            spike_idx += spike_delta

            target = [0, 1]
            if spike_idx >= (features // 2):
                target = [1, 0]

            yield spikes, torch.Tensor([target])

    return generator()


def _spikes_from_cfg(cfg):
    spike_sets = []
    for pattern in cfg['input']['patterns']:
        spike_sets.append(_gen_spikes(pattern, cfg['input_features']))

    return spike_sets


def _load_config(path):
    path = pathlib.Path(path)

    if not path.exists():
        raise ValueError("Path: {} doesn't exist".format(str(path)))

    with open(str(path), 'rb') as f:
        cfg = yaml.load(f)

    return cfg

    
def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--iters', type=int, default=100)
    
    return parser.parse_args()


def _eval(data, model, cfg, iters=None):
    correct = []
    model.stop_monitoring()
    for i, (target, spike_vector) in enumerate(data):
        model.reset_state()

        if i >= iters:
            break

        model(spike_vector)
        z = model(spike_vector)
        print("z: {}, target: {}".format(z, target))
        correct.append(z == target)

    correct = torch.vstack(correct)*1.0
    print("accuracy: ", correct.mean())
    model.resume_monitoring()


def _pack_spike_samples(spike_samples, padding=10):
    for sample in spike_samples:
        for i in range(padding):
            yield torch.zeros_like(sample)

        yield sample


def _gen_seq_pattern(cfg):
    size = cfg['data']['width']
    targets = cfg['data']['num_targets']
    backgrounds = cfg['data']['num_background']
    steps_per_pattern = cfg['data']['steps_per_pattern']
    steps = (targets + backgrounds)*steps_per_pattern
    spike_rate_range = cfg['data']['spike_rate_range']

    # Generate spikes for the whole simulation time-line
    p_rates = torch.rand((steps, size))
    p_rates = p_rates * abs(spike_rate_range[0] - spike_rate_range[1]) + min(spike_rate_range)
    full_spike_train = torch.poisson(p_rates).type(torch.int)

    # Pick out patterns
    pattern_indicies = torch.Tensor(range(targets+backgrounds))[torch.randperm(size)][0:targets]

    # Split timeline into segments
    spike_train_segments = [full_spike_train[i:i+steps_per_pattern] for i in range(0, steps, steps_per_pattern)]

    while True:
        rand_segment = torch.randint(0, len(spike_train_segments), (1,)).numpy()[0]
        segment = spike_train_segments[rand_segment]

        for time_slice in segment:
            time_slice = (time_slice > 0).type(torch.float)
            target = torch.tensor(int((rand_segment in pattern_indicies) * 1)).type(torch.float)
            yield target, time_slice
    


def _gen_onehot_pattern(cfg):
    width = cfg['data']['width']
    target_ratio = cfg['data']['target_ratio']

    # Generate a random set of one-hot vectors
    random_idx = torch.randperm(width)
    spike_vectors = torch.eye(width)[random_idx]

    # Pick width*target_ratio target patterns
    target_idxs = random_idx[range(0, int(width*target_ratio))]

    while True:
        # pick a random sample
        rand_idx = int(torch.randint(0, spike_vectors.shape[0], (1,)))

        spike_vector = spike_vectors[rand_idx].type(torch.float)
        target = torch.Tensor([rand_idx in target_idxs * 1.0]).type(torch.float)

        yield target, spike_vector
                

def _get_data(cfg):
    # Handle generated data
    def _get_gen_data(cfg):
        gen_data_functions = {
            'sequence': _gen_seq_pattern,
            'onehot': _gen_onehot_pattern
        }

        return gen_data_functions[cfg['data']['gen']](cfg)

    if 'gen' in cfg['data']:
        return _get_gen_data(cfg)


def _train(model, cfg, data, optim, reward_fn, monitors=[]):
    print("\nTraining")
    # Train
    for epoch in range(cfg['learning']['epochs']):
        for i, (target, spike_vector) in enumerate(data):
            # If samples are statistically independant, reset state
            if not cfg['learning']['correlated_inputs']:
                model.reset_state()
                optim.reset_state()

            if i >= cfg['learning']['iters_per_epoch']:
                break
            
            z = model(spike_vector, optimize=cfg['learning']['optimize'])
            z = model(torch.zeros_like(spike_vector), optimize=cfg['learning']['optimize'])
            reward = reward_fn(target, z)
            optim.step(reward=reward)

    print("\nAfter Training")
    model.reset_state()
    _eval(data, model, cfg, iters=cfg['learning']['iters_per_epoch'])
    

def _fill_registry():
    def _onehot_reward(target, z):
        reward = ((z == target)*1.0 - 0.5)*2.0

        return reward

    def _seq_reward(target, z):
        if target == z:
            return 1.0
        elif target != z and np.isclose(z, 0.0):
            return -1.0
        else:
            return 0.0
        
    registry.add_entry("_onehot_reward", _onehot_reward)
    registry.add_entry("_seq_reward", _seq_reward)
    
    
def _main():
    args = _parse_args()

    _fill_registry()
    cfg = _load_config(args.config)

    spike_monitor = logger.SpikeMonitor(weight_hist_iter=args.iters // 10)
    stdp_monitor = logger.STDPMonitor()
    stdp_optimizer = STDPOptimizer(alpha=1e-1, decay_fn='recent', monitor=stdp_monitor)

    model = PatternNet(cfg, optimizer=stdp_optimizer)
    model.add_monitor(spike_monitor)

    data = _get_data(cfg)
    # data_tmp = []
    # for i, (target, spike_vector) in enumerate(data):
    #     data_tmp.append((target, spike_vector))
        
    #     if i >= cfg['learning']['iters_per_epoch']:
    #         break
    # data = data_tmp

    reward_fn = registry.get_entry(cfg['learning']['reward_fn'])

    _train(model, cfg, data, stdp_optimizer, reward_fn)


if __name__ == '__main__':
    path = './runs/pattern'
    
    logger.set_path(path)
    
    with torch.no_grad():
        _main()
