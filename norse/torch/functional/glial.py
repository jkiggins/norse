from typing import Tuple

import torch

class AstroState:
    """
    State of astrocyte
    """

    def __init__(self, z: torch.Tensor):
        self.t_post = torch.zeros(z.shape, device=z.device, dtype=z.dtype)

    def decay(
            self,
            z_post: torch.Tensor,
            tau_post_inv: torch.Tensor,
            a_post: torch.Tensor,
            dt: float = 0.001,
    ):
        """
        Compute a decaying trace of activity based on z
        """
        
        self.t_post = self.t_post + (
            dt * tau_post_inv * (-self.t_post + a_post * z_post)
        )


class AstroParameters:
    """ Astrocyte Parameters """
