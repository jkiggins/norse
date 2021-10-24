import oyaml as yaml
import pathlib
import copy
from collections import OrderedDict

from pprint import pformat


import torch
class ConfigValue:
    def __init__(self, val):
         self.val = val

    def __call__(self):
        return self.val


class ConfigVariation(ConfigValue):
    def __init__(self, gen):
        super(ConfigVariation, self).__init__(None)

        # Store generator (or iterator) results
        self.seq = [x for x in gen]
        self.i = 0

        self.val = self.seq[self.i]


    def __iter__(self):
        return self.seq

    def next(self):
        self.i += 1

        rollover = False
        if self.i >= len(self.seq):
            rollover = True
            self.i = 0

        self.val = self.seq[self.i]
        
        return self(), rollover



class Config:
    def __init__(self, cfg_dict):
        self.cfg_dict = OrderedDict()
        for key in cfg_dict:
            val = cfg_dict[key]
            val = ConfigValue(val)
            if type(val()) == dict:
                val = Config(val())
            elif type(val()) == str:
                var = Config._parse_variation(val())
                if not (var is None):
                    val = var

            self.cfg_dict[key] = val

    def __iter__(self):
        for key in self.keys():
            yield key


    def __getitem__(self, key):
        val = self.cfg_dict[key]

        if type(val) in [ConfigValue, ConfigVariation]:
            return val()

        return val

    
    def __setitem__(self, key, val):
        if type(val) == dict:
            self.cfg_dict[key] = Config(val)
        else:
            self.cfg_dict[key] = ConfigValue(val)


    def __repr__(self):
        return pformat(self.as_dict())


    def __call__(self):
        return self.as_dict()


    def deref(self, name):
        key = self[name]
        return self[key]
        
        
    def as_dict(self, ordered=False):
        if ordered:
            repr_dict = OrderedDict()
        else:
            repr_dict = {}
        
        for key in self:
            if type(self[key]) == Config:
                repr_dict[key] = self[key].as_dict(ordered=ordered)
            else:
                repr_dict[key] = self[key]

        return repr_dict
            

    def _parse_variation(val):
        if val[0:8] == "linspace":
            params = val.split("linspace")[1]
            params = params.split('(')[1]
            params = params.split(')')[0]
            params = [float(s) for s in params.split(',')]

            params[2] = int(params[2])

            return ConfigVariation(torch.linspace(*params).tolist())


    def keys(self):
        return self.cfg_dict.keys()


    def traverse(self, bredth=True, objects=False):
        _next = [self]

        while len(_next) > 0:
            next_cfg = _next.pop(0)

            for key in sorted(next_cfg.keys()):
                if type(next_cfg[key]) == Config:
                    # Add to _next
                    if bredth:
                        _next.append(next_cfg[key])
                    else:
                        _next.insert(0,  next_cfg[key])

                # If next_item isn't a Config, return it
                elif objects:
                    yield next_cfg.cfg_dict[key]
                else:
                    yield next_cfg[key]
                    


    def apply(self, overlay):
        _next = [(overlay, self)]

        while len(_next) > 0:
            overlay, cfg = _next.pop(0)
            for key in overlay:
                if key in cfg:
                    if type(cfg[key]) == Config:
                        _next.append((overlay[key], cfg[key]))
                    else:
                        cfg[key] = overlay[key]
        
        
def load_config(path):
    path = pathlib.Path(path)

    if not path.exists():
        raise ValueError("Path: {} doesn't exist".format(str(path)))

    with open(str(path), 'rb') as f:
        cfg = yaml.safe_load(f)

    return Config(cfg)


def iter_variations(cfg, strategy):
    # Make list of ConfigVariation objects in cfg
    variations = []
    for node in cfg.traverse(bredth=True, objects=True):
        if type(node) == ConfigVariation:
            variations.append(node)

    stop = False
    while not stop:
        yield cfg

        if len(variations) == 0:
            break
        
        for i, node in enumerate(variations):
            if strategy == 'seq':
                _, rollover = node.next()
                
            elif strategy == 'foreach':
                _, rollover = node.next()
                if not rollover:
                    break

            # When the final ConfigVariation rolls over we are done
            if rollover and i == (len(variations) - 1):
                stop = True
                break


def iter_configs(cfg, exp_cfg):
    overlay = exp_cfg['overlay']

    for v in iter_variations(overlay, exp_cfg['variation_strategy']):
        cfg.apply(v)

        yield cfg
