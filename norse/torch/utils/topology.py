def _parse_node_name(name):
    if "." in name:
        node_name, node_index = name.split(".")
        node_index = str(node_index)

        return node_name, node_index

    return name, None
    
    

def connect_nodes(topology, nodes_dict):
    for k, v in topology.item():
        node = nodes_dict[k]
        
        for next_node_pad in v['next']:
            next_node_name = _node_name(next_node_pad)
            next_node = nodes_dict[next_node_name]

            node.connect_to(next_node_pad, next_node)

            
            
class Node:
    def __init__(self, name, num_inputs, num_outputs, obj):
        self.obj = obj
        self.name = name

        self.connections = {}

        self._input_pads = {"{}.{}".format(name, i): None for i in range[num_inputs]}
        self._input_pads[name] = None
        
        self._output_pads = {"{}.{}".format(name, i): None for i in range[num_outputs]}
        self._output_pads[name] = None
        

    def connect(self, local_pad, remote_pad, node):
        if not(node in self.connections):
            self.connections[node] = []

        if not ((local_pad, remote_pad) in self.connections[node]):
            self.connections[node].append((local_pad, remote_pad))


    def _set_pad_value(self, name, value, pad_value_dict, strategy='sum'):
        if not (name in pad_value_dict)):
            raise ValueError("{} not in pad value dict".format(name))

        new_val = value        
        if not (self.pad_value_dict[name] is None):
            if strategy == 'sum':
                new_value = self.pad_value_dict[name] + value

        self.pad_value_dict[name] = new_value

        
    def set_output_pad_value(self, name, value, strategy="sum"):
        self._set_pad_value(name, value, self._output_pads, strategy=strategy) 
        

    def set_input_pad_value(self, name, value, strategy='sum'):
        self._set_pad_value(name, value, self._input_pads, strategy=strategy)


    def get_input_pad_value(self, name):
        if not(name in self._input_pads):
            raise ValueError("{} is not a pad".format(name))
        
        return self._input_pads[name]

    
    def forward(self, strategy='sum'):
        """
        Forward values along pad (node1[output] -> node2[input]) connections
        """
        for node, pad_pairs in self.connections.items():
            for pad_pair in pad_pairs:
                local_pad, remote_pad = pad_pair

                local_pad_val = self.get_output_pad_value(local_pad)
                node.set_input_pad_value(remote_pad, local_pad_value, strategy=strategy)
        

    def nodes():
        return list(self.connections.keys())
