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
    
