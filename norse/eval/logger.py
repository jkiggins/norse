import os
from torch.utils.tensorboard import SummaryWriter

import torch

_writer_ = None

def set_path(path):
    global _writer_
    _writer_ = SummaryWriter(log_dir=os.path.abspath(path))


def writer():
    global _writer_
    return _writer_


def _iter_count(dict, name):
        if not(name in dict):
            dict[name] = 0
        dict[name] += 1

        return dict[name] - 1


class SpikeMonitor:
    def __init__(self, weight_hist_iter=None):
        self.iter_dict = {}
        self.weight_hist_iter = weight_hist_iter
        
    def reset():
        self.iter_dict = {}

    def iter_scalar(self, name, scalar):
        index = _iter_count(self.iter_dict, name)
        writer().add_scalar(name, scalar, index)

    def iter_weights(self, name, weights):
        index = _iter_count(self.iter_dict, name)

        if index % self.weight_hist_iter == 0:
            writer().add_histogram(name, weights.squeeze(), index // self.weight_hist_iter) 
            
        
    def __call__(self, pre, lif_state, weight_module, spiking_module, post, pre_is_input=False, name=None):
        avg_spikes = post.mean()

        if lif_state:
            avg_membrane_voltage = lif_state.v.mean()
        else:
            avg_membrane_voltage = 0.0

        name = "{}_spike_activity".format(name)
        self.iter_scalar(name, avg_spikes)

        name = "{}_membrane_voltage".format(name)
        self.iter_scalar(name, avg_membrane_voltage)

        name = "{}_hist".format(name)
        self.iter_weights(name, weight_module.weight.data)

        if pre_is_input:
            name = "input_spike_activity"
            self.iter_scalar(name, pre.mean())



class STDPMonitor:
    def __init__(self):
        self.iter_dict = {}
        
    def __call__(self, module, pre, post, w, dw, name=None):
        avg_dw = dw.mean()

        name = "{}_dw".format(name)
        index = _iter_count(self.iter_dict, name)
        writer().add_scalar(name, avg_dw, index)


class TraceLogger:
    def __init__(self):
        self.traces = {}
        self.step = 0
        self.last_mean = {}

    def step_trace(self, name, value):
        if not (name in self.traces):
            self.traces[name] = []
            
        self.traces[name].append(value)

    def set_trace(self, name, value):
        if not (name in self.traces):
            self.traces[name] = []

        self.traces[name] = value


    def apply(self, fn):
        for trace in self.traces:
            self.traces[trace] = fn(self.traces[trace])


    def log_traces(self, mean_delta=None):
        for trace, data in self.traces.items():
            if type(data) == list:
                data = torch.Tensor(data)
            
            if mean_delta:
                if not (trace in self.last_mean):
                    self.last_mean[trace] = None

                last_mean = self.last_mean[trace]
            
                mean_change = 1.0
                if last_mean:
                    mean_change = abs(data.mean() - last_mean) / last_mean
                self.last_mean[trace] = data.mean()

                if mean_change < mean_delta:
                    continue
                
            print("Logging Spike Deltas")
            writer().add_histogram(trace, data, global_step=self.step)
            self.step += 1
                
        self.traces.clear()
    
