import torch
from torch.optim import Optimizer

class AlexRicciNavier(Optimizer):
    """
    Alex-Ricci-Navier (ARN) Optimizer
    A physics-inspired optimizer combining Navier-Stokes viscosity 
    and Ricci Flow curvature for gradient stabilization.
    """
    def __init__(self, params, lr=1e-3, nu=0.1, kappa=0.01, betas=(0.9, 0.999), eps=1e-8):
        # nu: Viscosity coefficient (Navier-Stokes)
        # kappa: Curvature coefficient (Ricci Flow)
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
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # --- ARN Core Logic ---
                # 1. Apply Viscosity (Stabilization)
                grad.add_(p, alpha=group['nu']) 

                # 2. Apply Curvature (Geometric adjustment)
                grad.add_(group['kappa'] * grad * torch.abs(grad)) 

                # --- Adaptive Momentum ---
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.sqrt() / bias_correction2).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
