# Torch imports
import torch
from torch import nn

from norse.torch.utils import plot as nplot
from matplotlib import pyplot as plt

# Norse Imports
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
from norse.torch.module.stdp import STDPOptimizer
from norse.eval import logger
from norse.torch.utils import registry, config
from norse.torch.module.astrocyte import Astrocyte
from norse.torch.functional.astrocyte import AstroState, AstroParams, AstroActivityParams

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



class AstroAdaptNet(torch.nn.Module):
    # TODO: Convert to from_cfg format
    def __init__(self, cfg,
                 lif_params,
                 linear_params,
                 optimizer=None,
                 optimize=True,
                 topology=None,
                 dt=0.001,
                 dtype=torch.float):

        super(AstroAdaptNet, self).__init__()

        dt = cfg['sim']['dt']
        
        self.optimizer = optimizer
        self.optimize = cfg['learning']['optimize']
        
        self.monitors = []

        top = topology

        self.linear = _linear_from_params(top['input']['num'], top['lif']['num'], linear_params)
        self.lif = _lif_from_params(lif_params)
        self.lif_state = None

        self.astro = Astrocyte.from_cfg(cfg)
        self.astro_state = None
        self.last_astro_effect = None


    def from_cfg(cfg, optimizer):
        top_key = cfg['topology']

        if not (top_key in cfg):
            raise ValueError("Topology {} not defined in config".format(top_key))
        top = cfg[top_key]

        return AstroAdaptNet(cfg,
                             lif_params=cfg['lif_params'],
                             linear_params=cfg['linear_params'],
                             optimizer=optimizer,
                             optimize=cfg['learning']['optimize'],
                             topology=top,
                             dt=cfg['sim']['dt'])


    def forward(self, z):
        z_pre = z

        z = self.linear(z)
        if self.last_astro_effect: z += self.last_astro_effect
        z, lif_state = self.lif(z, self.lif_state)

        effect, astro_state = self.astro.forward(z, self.astro_state)
        
        self.last_astro_effect = effect
        
        self.astro_state = astro_state
        self.lif_state = lif_state

        # if self.optimize: self.optimizer(self.linear, z_pre, z)

        return z, lif_state, astro_state, effect


def _graph_sim(cfg, monitor):
    fig = plt.Figure(figsize=(10, 15))

    subplot_shape = (7,1)
    
    # Graph timelines
    ax = fig.add_subplot(*subplot_shape,1)
    ax.set_title("Presynaptic Spike Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spikes")
    trace = monitor.trace("pre")
    trace = torch.stack(trace)
    nplot.plot_spikes_2d(trace, axes=ax)

    ax = fig.add_subplot(*subplot_shape,2)
    ax.set_title("Postsynaptic Spike Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spikes")
    trace = monitor.trace("post")
    trace = torch.stack(trace)
    nplot.plot_spikes_2d(trace, axes=ax)

    ax = fig.add_subplot(*subplot_shape,3)
    ax.set_title("Astrocyte State Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    trace = monitor.trace("astro_state")
    ax.plot(trace)

    ax = fig.add_subplot(*subplot_shape, 4)
    ax.set_title("Astrocyte Effect Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("Astrocyte Effect")
    trace = monitor.trace("astro_effect")
    ax.plot(trace)

    ax = fig.add_subplot(*subplot_shape, 5)
    ax.set_title("Presynaptic Moving Average")
    ax.set_xlabel("Time")
    ax.set_ylabel("Moving Average Firing Rate")
    trace = monitor.trace("pre_avg_rate")
    ax.plot(trace)

    ax = fig.add_subplot(*subplot_shape, 6)
    ax.set_title("Postsynpatic Moving Average")
    ax.set_xlabel("Time")
    ax.set_ylabel("Moving Average Firing Rate")
    trace = monitor.trace("post_avg_rate")
    ax.plot(trace)

    ax = fig.add_subplot(*subplot_shape, 7)
    ax.set_title("Pre and Postsynpatic FFT")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Intensity")
    trace = monitor.trace("post", as_tensor=True)
    trace_fft = torch.fft.rfft(trace)
    ax.plot(trace_fft)
    trace = monitor.trace("pre", as_tensor=True)
    trace_fft = torch.fft.rfft(trace)
    ax.plot(trace_fft)
    ax.legend()

    fig.subplots_adjust(left=0.1,
                        bottom=0.4, 
                        right=0.9, 
                        top=1.4, 
                        wspace=0.4, 
                        hspace=0.4)

    path = pathlib.Path(logger.get_path())/"{}.png".format(cfg['sim']['sim_name'])
    fig.savefig(str(path), bbox_inches="tight")


def _graph_experiment(cfg, monitor):
    fig = plt.Figure()

    ax = fig.add_subplot(2,1,1)
    ax.set_title("Pre and Post Synaptic Volume vs. Input Frequency")
    ax.set_xlabel("Input Poisson Rate")
    ax.set_ylabel("Spiking Volume")
    
    pre_trace = monitor.trace("pre_volume")
    rate_trace = monitor.trace("rate")
    ax.plot(rate_trace, pre_trace, label="Pre")

    post_trace = monitor.trace("post_volume")
    ax.plot(rate_trace, post_trace, label="Post")


    ax = fig.add_subplot(2,1,2)
    ax.set_title("Pre and Post Synaptic Volume vs. Tau")
    ax.set_xlabel("Tau")
    ax.set_ylabel("Spiking Volume")
    
    pre_trace = monitor.trace("pre_volume")
    tau_trace = monitor.trace("tau")
    ax.plot(tau_trace, pre_trace, label="Pre")

    post_trace = monitor.trace("post_volume")
    ax.plot(tau_trace, post_trace, label="Post")


    path = pathlib.Path(logger.get_path())/"{}.png".format(cfg['sim']['exp_name'])

    fig.legend()
    fig.savefig(str(path), bbox_inches="tight")

    


def _sim(cfg, data, model, monitor, optimizer=None):

    # Clear monitor
    monitor.clear("pre")
    monitor.clear("post")
    monitor.clear("astro_state")
    monitor.clear("astro_effect")
    monitor.clear("weight")
    
    model.optimizer = optimizer
    
    for i, spike_vector in enumerate(data):
        if i > cfg['sim']['iters']:
            break

        z, lif_state, astro_state, astro_effect = model(spike_vector)
        # if optimizer: optimizer.step()

        # Graph a timeline of intermediate steps
        monitor("pre", spike_vector)
        monitor("post", z)
        monitor("astro_state", astro_state.t_z)
        monitor("astro_effect", astro_effect)
        monitor("weight", float(model.linear.weight.data.view(-1)))

    monitor.moving_average('pre', 'pre_avg_rate', window=50)
    monitor.moving_average('post', 'post_avg_rate', window=50)

    _graph_sim(cfg, monitor)
    
    
def _experiment(cfg, exp_cfg, name=None):
    monitor = logger.TraceLogger()
    
    for i, next_cfg in enumerate(config.iter_configs(cfg, exp_cfg)):
        next_cfg['sim']['sim_name'] = _sim_name_from_cfg(next_cfg, i)
        next_cfg['sim']['exp_name'] = _exp_name_from_cfg(next_cfg, name)

        print(yaml.dump(next_cfg.as_dict()))

        data = _get_data(next_cfg)
        stdp_optimizer = STDPOptimizer.from_cfg(next_cfg)
        model = AstroAdaptNet.from_cfg(next_cfg, stdp_optimizer)

        _sim(cfg, data, model, monitor, optimizer=stdp_optimizer)

        # Aggregate stats
        pre_trace = monitor.trace('pre', as_tensor=True)
        post_trace = monitor.trace('post', as_tensor=True)
        monitor('post_volume', post_trace.sum())
        monitor('pre_volume', pre_trace.sum())

        monitor('rate', next_cfg['data']['stops'])
        monitor('tau', next_cfg['astro_params']['tau'])

    _graph_experiment(cfg, monitor)

        
def _gen_poisson_ramp(cfg):

    width = cfg['data']['width']
    stops = cfg['data']['stops']
    steps = cfg['data']['steps']

    if not hasattr(stops, '__iter__'):
        stops = [stops]
    
    if len(stops) == 0:
        raise ValueError("At least one stop is needed for poisson ramp")
    elif len(stops) == 1:
        rate_seq = stops
    else:
        rate_seq = torch.Tensor([])
        steps_per_stop = int(steps / (len(stops) - 1))
        
        for i in range(len(stops)-1):
            start = stops[i]
            stop = stops[i+1]
            
            rate_seq = torch.cat((rate_seq, torch.linspace(start, stop, steps_per_stop+1)))

    while True:
        for rate in rate_seq:
            p_rate = torch.ones((width)) * rate

            spike_vector = torch.poisson(torch.as_tensor(p_rate)).view(1)
            
            yield (spike_vector > 0.1)*1.0
    

def _get_data(cfg):
    # Handle generated data
    def _get_gen_data(cfg):
        gen_data_functions = {
            'poisson_ramp': _gen_poisson_ramp,
        }

        return gen_data_functions[cfg['data']['gen']](cfg)

    if 'gen' in cfg['data']:
        return _get_gen_data(cfg)


def _sim_name_from_cfg(cfg, i):
    activity_config = cfg['astro_params'].deref('activity_params')
    target = activity_config['target']

    astro_tau = cfg['astro_params']['tau']
    astro_alpha = cfg['astro_params']['alpha']
    
    name = "a{:07.3f}_t{:07.3f}_tar{:07.3f}-{:04d}".format(astro_alpha, astro_tau, target, i)

    return name


def _exp_name_from_cfg(cfg, exp_name):
    return exp_name
        

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    
    return parser.parse_args()



def _main():
    args = _parse_args()

    logger.set_path('./runs/astro_test')

    cfg = config.load_config(args.config)
    
    with torch.no_grad():
        for exp_name in cfg['meta']['experiments']:
            _experiment(cfg, cfg['meta'][exp_name], name=exp_name)
        
        
if __name__ == '__main__':
    _main()
