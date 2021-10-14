import torch

def _get_pad_name(pad_idx, name, template=None):
    return template.format(name, pad_idx)

        
class Node:
    def __init__(self, num_inputs, num_outputs, name="", template="{}.{}"):
        self.input_pads(num_inputs, name=name, template=template)
        self.output_pads(num_outputs, name=name, template=template)

        self.out_conns = {}
        self.in_conns = {}


    def input_pads(self, num_pads, name="", template="{}.{}"):
        self.in_pads = torch.zeros((num_pads))
        self.in_pad_names = {_get_pad_name(name, i): i for i in range(num_pads)}


    def output_pads(self, num_pads, name="", template="{}.{}"):
        self.out_pads = torch.zeros((num_pads))
        self.out_pad_names = {_get_pad_name(name, i): i for i in range(num_pads)}


    def connect_out(self, src_pad_name, dest, dest_pad_name):
        if not (dest in self.out_conns):
            self.out_conns[dest] = []

        pad_conn = (src_pad_name, dest_pad_name)

        if not (pad_conn in self.out_conns[dest]):
            self.out_conns.append(pad_conn)
            dest.connect_in(dest_pad_name, self, src_pad_name)


    def connect_in(self, local_pad_name, dest, dest_pad_name):
        if not (dest in self.in_conns):
            self.in_conns[dest] = []

        pad_conn = (local_pad_name, dest_pad_name)

        if not (pad_conn in self.in_conns[dest]):
            self.in_conns.append(pad_conn)
            dest.connect_out(dest_pad_name, self, local_pad_name)

        
    def forward(self):
        self.output_pads = self.input_pads.clone()
