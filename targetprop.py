from enum import Enum, unique

import torch


def sign11(Z):
    H = torch.sign(Z).clamp(min=0) * 2 - 1
    return H


def soft_hinge(Z, T):
    loss = torch.tanh(-(Z * T).float()) + 1
    return loss


@unique
class TPRule(Enum):
    SSTE = 2
    SoftHinge = 5

    @staticmethod
    def sste_backward(Z, dH, T, a=0, b=1):
        dZ = dH * torch.ge(Z, a).float() * torch.le(Z, b).float()
        return dZ

    @staticmethod
    def softhinge_backward(Z, dH, T):
        if T is None:
            T = torch.sign(-dH)
        z = soft_hinge(Z, T) - 1
        dZ = (1 - z * z) * -T
        return dZ

    @staticmethod
    def get_backward_func(targetprop_rule):
        if targetprop_rule == TPRule.SSTE:
            tp_grad_func = TPRule.sste_backward
        elif targetprop_rule == TPRule.SoftHinge:
            tp_grad_func = TPRule.softhinge_backward
        else:
            raise ValueError('specified targetprop rule ({}) has no backward function'.format(targetprop_rule))
        return tp_grad_func
