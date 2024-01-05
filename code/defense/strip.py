import torch
import numpy as np
from tqdm import tqdm
import random

class STRIP():

    def __init__(self, args, inspection_set, clean_set, model, strip_alpha: float = 0.5, N: int = 64, defense_fpr: float = 0.05, batch_size=128, alpha=0.2):

        self.args = args

        self.strip_alpha: float = strip_alpha
        self.N: int = N
        self.defense_fpr = defense_fpr

        self.inspection_set = inspection_set
        length = int(len(clean_set) * alpha)
        self.clean_set, _ = torch.utils.data.random_split(clean_set, [length, len(clean_set)-length])

        self.model = model
        self.batch_size = batch_size


    def cleanse(self):

        # choose a decision boundary with the test set
        clean_entropy = []
        clean_set_loader = torch.utils.data.DataLoader(self.clean_set, batch_size=self.batch_size, shuffle=False)
        for _input, _label in tqdm(clean_set_loader):
            _input, _label = _input.cuda(), _label.cuda()
            entropies = self.check(_input, _label, self.clean_set)
            for e in entropies:
                clean_entropy.append(e)
        clean_entropy = torch.FloatTensor(clean_entropy)

        clean_entropy, _ = clean_entropy.sort()
        threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_entropy))])
        threshold_high = np.inf

        inspection_set_loader = torch.utils.data.DataLoader(self.inspection_set, batch_size=self.batch_size, shuffle=False)
        all_entropy = []
        for _input, _label in tqdm(inspection_set_loader):
            _input, _label = _input.cuda(), _label.cuda()
            entropies = self.check(_input, _label, self.clean_set)
            for e in entropies:
                all_entropy.append(e)
        all_entropy = torch.FloatTensor(all_entropy)

        suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
        return suspicious_indices

    def check(self, _input: torch.Tensor, _label: torch.Tensor, source_set) -> torch.Tensor:
        _list = []

        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:self.N]

        with torch.no_grad():

            for i in samples:
                X, _ = source_set[i]
                X = X.cuda()
                _test = self.superimpose(_input, X)
                entropy = self.entropy(_test).cpu().detach()
                _list.append(entropy)

        return torch.stack(_list).mean(0)

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.strip_alpha

        result = _input1 + alpha * _input2

        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        p = torch.nn.Softmax(dim=1)(self.model(_input)) + 1e-8
        return (-p * p.log()).sum(1)

def cleanser(inspection_set, clean_set, model, args):

    worker = STRIP( args, inspection_set, clean_set, model, strip_alpha=1.0, N=100, defense_fpr=0.1, batch_size=128)
    suspicious_indices = worker.cleanse()

    return suspicious_indices