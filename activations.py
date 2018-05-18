import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import targetprop as tp


class Sign11F(Function):
    ''''''

    def __init__(self, targetprop_rule=tp.TPRule.SoftHinge):
        super(Sign11F, self).__init__()
        self.tp_rule = targetprop_rule
        self.target = None

    def forward(self, Z):
        self.save_for_backward(Z)
        H = tp.sign11(Z)
        return H

    def backward(self, dH):
        Z, = self.saved_tensors
        dZ = None
        if self.needs_input_grad[0]:
            tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
            if self.tp_rule == tp.TPRule.SSTE:
                dZ = tp_grad_func(Z, dH, self.target, a=-1, b=1)
            elif self.tp_rule == tp.TPRule.SoftHinge:
                dZ = tp_grad_func(Z, dH, self.target) * dH.abs()
        return dZ


class Sign11(nn.Module):
    ''''''

    def __init__(self, targetprop_rule=tp.TPRule.SoftHinge):
        super(Sign11, self).__init__()
        self.tp_rule = targetprop_rule

    def __repr__(self):
        s = '{name}(a={a}, b={b}, tp={tp})'
        return s.format(name=self.__class__.__name__, a=-1, b=1, tp=self.tp_rule)

    def forward(self, Z):
        H = Sign11F(targetprop_rule=self.tp_rule)(Z)
        return H


class qReLUF(Function):
    ''''''

    def __init__(self, targetprop_rule=tp.TPRule.SoftHinge, nsteps=3):
        super(qReLUF, self).__init__()
        assert nsteps >= 1
        assert nsteps < 255
        self.tp_rule = targetprop_rule
        self.nsteps = nsteps
        self.target = None

    def forward(self, Z):
        self.save_for_backward(Z)
        H = Z * (self.nsteps - 1)
        H.ceil_().clamp_(min=0, max=self.nsteps)
        H = H * (1.0 / self.nsteps)
        return H

    def backward(self, dH):
        Z, = self.saved_tensors
        dZ = None
        if self.needs_input_grad[0]:
            tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
            if self.tp_rule == tp.TPRule.SSTE:
                dZ = tp_grad_func(Z, dH, self.target, a=0, b=1)
            elif self.tp_rule == tp.TPRule.SoftHinge:
                dZ = tp_grad_func(Z, dH, self.target) * dH.abs()
        return dZ


class qReLU(nn.Module):
    ''''''

    def __init__(self, targetprop_rule=tp.TPRule.SoftHinge, nsteps=3):
        super(qReLU, self).__init__()
        self.tp_rule = targetprop_rule
        self.nsteps = nsteps

    def forward(self, Z):
        H = qReLUF(targetprop_rule=self.tp_rule, nsteps=self.nsteps)(Z)
        return H

    def __repr__(self):
        s = '{}(steps={})'
        return s.format(self.__class__.__name__, self.nsteps)


class ThresholdReLU(nn.Module):
    ''''''

    def __init__(self, max_val=1., slope=1.):
        super(ThresholdReLU, self).__init__()
        self.max_val = max_val
        self.slope = slope

    def forward(self, Z):
        return F.relu(Z * self.slope if self.slope != 1 else Z).clamp(max=self.max_val)
