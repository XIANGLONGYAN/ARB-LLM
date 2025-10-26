import math
import time
from exceptiongroup import catch
import torch
import torch.nn as nn
import transformers
from utils.structure_arb import structural_guassian_distribution_multip_alternating_group_x

import logging
logger = logging.getLogger()

DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

'''
BRAGPTQ is the meaning of GPTQ used Binary Residual Approximation in paper to realize 1-bit quantization
BRAGPTQ uses structural mask to distinguish outliers and other data, and takes advantage of part of GPTQ to lower error
'''
class BRAGPTQ:
    def __init__(
        self, layer, braq_quantizer,salient_metric, disable_gptq=False, method='arb', order2_group=False
    ):
        self.method = method
        self.order2_group = order2_group
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq

        self.inp = []
        # self.inp2 = torch.zeros(self.columns, self.columns, device=self.dev, dtype=torch.float16)

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        # save memory
        # print(inp.shape)   # [1,2048,4096]
        # print(inp[0].T.shape)
        # self.inp2 = self.inp2 + (inp[0].T @ inp[0])

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            
        if self.method == 'arb-x':
            self.inp.append(inp)
            
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01,
                    orders=(1,1,2),
                    num_p=1,
                    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        if self.method == 'arb-x':
            self.inp = torch.concat(self.inp)
        # print(self.inp.shape)

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed

            if self.method == 'arb-x':
                # S = torch.einsum('bki,bkj->ij', self.inp[:, :, st:ed], self.inp[:, :, st:ed])
                S = torch.matmul(self.inp[:, :, st:ed].to(torch.float32).transpose(1, 2), self.inp[:, :, st:ed].to(torch.float32)).mean(dim=0)   # avoid overflow
            else:
                S = None
            # S = self.inp2[st:ed, st:ed]
            # print(S==S2)

            if self.order2_group:
                num_mask = 2 * (num_p+1)
                orders = [2 for _ in range(num_p+1)] + [1 for _ in range(num_p+1)]
            else:
                num_mask = 1 + num_p + 1
                orders = [2] + [1 for _ in range(num_p+1)]
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(num_mask, dim=0)
            mask_list = structural_guassian_distribution_multip_alternating_group_x(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50, num_p, S, self.method, self.order2_group)
            for i in range(num_mask):
                mask[i] = mask_list[i]

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                
                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                # old_q_part_groups = []
                q_part_groups = []

                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i], S=S))

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2
                    # breakpoint()

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    x_error = torch.sum((self.layer(self.inp1) - self.out1) ** 2)
                    # print(torch.sum(Losses))

        torch.cuda.synchronize()
        # print("time %.2f" % (time.time() - tick))
        # print("error", torch.sum(Losses).item())
        times = time.time() - tick
        logger.info(f'time {times:.2f}')
        logger.info(f'error {torch.sum(Losses).item()}')
        logger.info(f'x error {x_error.item()}')

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        # del mask1, mask2, mask3
        del mask_list
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv, self.inp, S, q_part_groups
        # del H, Hinv, self.inp2, S, q_part_groups
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
