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

    def __repr__(self):
        return "ConfigValue({})".format(self.val)


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


    def __repr__(self):
        return "ConfigVariation({})".format(self.val)



class Config:
    def __init__(self, cfg_dict):        
        self.cfg_dict = OrderedDict()

        # if '__inherit__' in cfg_dict:
        #     inherit_cfg = Config(cfg_dict['__inherit__'])
        #     self.apply(inherit_cfg)
        #     del cfg_dict['__inherit__']

        
        for key in cfg_dict:
            val = cfg_dict[key]

            # If the val is a dict, make a config out of it, then apply to current value at that key (if present)
            if Config._is_like_dict(val):
                val = Config(val)
                if not (key in self.cfg_dict):
                    self.cfg_dict[key] = Config({})
                    
                self.cfg_dict[key].apply(val, objects=True)

            # If the val is a string, check if it is a variation string, save otherwise
            elif type(val) == str:
                var = Config._parse_variation(val)
                if not (var is None):
                    val = var
                
                self.cfg_dict[key] = val

            # It's just a value, save it
            else:
                self.cfg_dict[key] = val

    def _is_like_dict(val):
        return type(val) in [dict, OrderedDict]

    def _is_like_value(val):
        return type(val) in [ConfigValue, ConfigVariation]


    def __iter__(self):
        for key in self.keys():
            yield key


    def __getitem__(self, key):
        val = self.cfg_dict[key]

        if Config._is_like_value(val):
            return val()

        return val

    def __copy__(self):
        return Config(self.as_dict(ordered=True))

    
    def __setitem__(self, key, val):
        if Config._is_like_dict(val):
            self.cfg_dict[key] = Config(val)
        elif type(val) == Config:
            self.cfg_dict[key] = val
        elif Config._is_like_value(val):
            self.cfg_dict[key] = val
        else:
            self.cfg_dict[key] = ConfigValue(val)


    def __repr__(self):
        cfg_dict = self.as_dict(objects=True)

        # return yaml.dump(cfg_dict)
        return pformat(cfg_dict)


    def __call__(self):
        return self.as_dict()


    def deref(self, name):
        key = self[name]
        return self[key]
        
        
    def as_dict(self, ordered=False, objects=False):
        if ordered:
            repr_dict = OrderedDict()
        else:
            repr_dict = {}
        
        for key in self:
            if type(self[key]) == Config:
                repr_dict[key] = self[key].as_dict(ordered=ordered, objects=objects)
            elif objects:
                repr_dict[key] = self.cfg_dict[key]
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


    def apply(self, overlay, objects=False):
        _next = [(overlay, self)]

        while len(_next) > 0:
            overlay, cfg = _next.pop(0)
            for key in overlay:
                if type(overlay[key]) == Config:
                    if not (key in cfg):
                        cfg[key] = {}
                    _next.append((overlay[key], cfg[key]))
                elif objects:
                    cfg[key] = overlay.cfg_dict[key]
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

    cfg = copy.deepcopy(cfg)

    for v in iter_variations(overlay, exp_cfg['variation_strategy']):
        cfg.apply(v)

        yield cfg
