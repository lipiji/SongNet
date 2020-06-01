import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, device, size, padding_idx, label_smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        self.size = size
        self.device = device

        self.smoothing_value = label_smoothing / (size - 2)
        self.one_hot = torch.full((1, size), self.smoothing_value).to(device)
        self.one_hot[0, self.padding_idx] = 0
        
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        real_size = output.size(1)
        if real_size > self.size:
            real_size -= self.size
        else:
            real_size = 0

        model_prob = self.one_hot.repeat(target.size(0), 1)
        if real_size > 0:
            ext_zeros = torch.full((model_prob.size(0), real_size), self.smoothing_value).to(self.device)
            model_prob = torch.cat((model_prob, ext_zeros), -1)
        model_prob.scatter_(1, target, self.confidence)
        model_prob.masked_fill_((target == self.padding_idx), 0.)

        return F.kl_div(output, model_prob, reduction='sum')
