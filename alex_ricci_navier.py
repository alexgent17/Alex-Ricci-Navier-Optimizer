# ==========================================================
# ALEX-RICCI-NAVIER (ARN) OPTIMIZER
# ==========================================================
# High-stability neural network optimizer based on 
# Navier-Stokes damping and Ricci Flow curvature.
#
# Key Features:
# - Fluid-dynamics inspired gradient smoothing.
# - Manifold curvature adjustment (Ricci Flow).
# - Robustness against exploding and unstable gradients.
#
# Developed by Alex.
# ==========================================================

import torch
from torch.optim import Optimizer

class AlexRicciNavier(Optimizer):
    def __init__(self, params, lr=1e-3, nu=0.1, kappa=0.01, betas=(0.9, 0.999), eps=1e-8):
        """
        ARN Optimizer
        :param lr: Learning rate.
        :param nu: Kinematic viscosity (Navier-Stokes damping).
        :param kappa: Ricci curvature scaling factor.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, nu=nu, kappa=kappa, betas=betas, eps=eps)
        super(AlexRicciNavier, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # 1. Navier-Stokes Damping (Viscosity)
                # Helps prevent sudden spikes in gradients
                grad.add_(p, alpha=group['nu'])

                # 2. Ricci Flow Adjustment
                # Adjusts the "geometry" of the weight space
                curvature_adj = group['kappa'] * grad * torch.abs(grad)
                grad.add_(curvature_adj)

                # 3. Momentum and Scaling (Adam-style baseline)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / (1 - beta2 ** state['step'])).add_(group['eps'])
                step_size = group['lr'] / (1 - beta1 ** state['step'])

                # Final Update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
