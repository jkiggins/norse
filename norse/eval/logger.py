import os
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

from norse.torch.utils import plot as nplot


_writer_ = None
_path_ = None

def set_path(path):
    global _writer_
    global _path_

    _path_ = os.path.abspath(path)
    _writer_ = SummaryWriter(log_dir=_path_)


def savefig(fig, file_name):
    global _path_
    
    path = pathlib.Path(_path_)/file_name
    fig.savefig(str(path))


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
        self.timeline_dict = {}
        self.weight_hist_iter = weight_hist_iter
        self.do_monitor = True
        self.timestep = 0
        
    def reset():
        self.iter_dict = {}

    def iter_scalar(self, name, scalar):
        index = _iter_count(self.iter_dict, name)
        writer().add_scalar(name, scalar, index)

    def iter_weights(self, name, weights):
        index = _iter_count(self.iter_dict, name)

        if index % self.weight_hist_iter == 0:
            writer().add_histogram(name, weights.squeeze(), index // self.weight_hist_iter)

            
    def timeline_step(self, name, spikes):
        if not(name in self.timeline_dict):
            self.timeline_dict[name] = []

        if len(self.timeline_dict[name]) > 0:
            if self.timeline_dict[name][-1].shape != spikes.shape:
                raise ValueError("Timeline tensors must be the same shape")
            
        self.timeline_dict[name].append(spikes)

        
    def timeline_tensor(self):
        tensor_set = []
        
        for name, timeline in self.timeline_dict.items():
            test_same_shape = all([timeline[0].shape == t.shape for t in timeline])
            if not test_same_shape:
                raise ValueError("Timeline tensors must be the same shape")

            tensor_set.append(torch.vstack(timeline))
            
        return torch.vstack(tensor_set)


    def graph_timeline(self, ax=None):
        timeline_2d = self.timeline_tensor()

        return nplot.plot_spikes_2d(timeline_2d, axes=ax)


    def stop(self):
        self.do_monitor = False

        
    def resume(self):
        self.do_monitor = True

        
    def __call__(self, pre, lif_state, weight_module, spiking_module, post, pre_is_input=False, name=None):


        if not self.do_monitor:
            self.timestep += 1
            return

        timestep_log = "Timestep {}:".format(self.timestep)
        
        
        # self.timeline_step(name, pre)
        if not (lif_state is None):
            timestep_log += "\nlif_state:"
            for i, v in enumerate(lif_state.v):
                timestep_log += "{}v ".format(v)
                iter_name = "l{}_n{}_membrane_voltage".format(name, i)
                self.iter_scalar(iter_name, v)

        timestep_log += "\noutput spikes: "
        for i, s in enumerate(post):
            timestep_log += "{} ".format(s)
            iter_name = "l{}_n{}_spike_activity".format(name, i)
            self.iter_scalar(iter_name, s)

        timestep_log += "\ninput spikes: "
        for s in pre:
            timestep_log += "{} ".format(s)

        iter_name = "{}_hist".format(name)
        self.iter_weights(iter_name, weight_module.weight.data)

        if pre_is_input:
            for i, s in enumerate(pre):
                iter_name = "input{}_spike_activity".format(i)
                self.iter_scalar(iter_name, s)

        print()
        print(timestep_log)
        print()
        self.timestep += 1



class STDPMonitor:
    def __init__(self):
        self.iter_dict = {}
        
    def __call__(self, module, pre, post, w, dw, name=None):
        for i in range(dw.shape[0]):
            for j in range(dw.shape[1]):
                iter_name = "{}_i{}_o{}_dw".format(name, j, i)
                index = _iter_count(self.iter_dict, iter_name)
                writer().add_scalar(iter_name, dw[i,j], index)


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
    
