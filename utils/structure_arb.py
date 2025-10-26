import torch
from utils.autosearch_arb import structural_searching_multip, structural_searching_multip_alternating_group, structural_searching_multip_alternating_group_x, structural_searching_multip_alternating_group_rc

import logging
logger = logging.getLogger()

'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution_multip_alternating_group_x(tmp, H=None, metric="magnitude", up_lim=30, num_p=1, inp=None, method='arb', order2_group=False):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # print(f'debug', inp)
    if method == 'arb':
        optimal_split_list, mask_list = structural_searching_multip_alternating_group(target_weights, up_lim, num_p, inp, order2_group=order2_group)
    elif method == 'arb-x':
        optimal_split_list, mask_list = structural_searching_multip_alternating_group_x(target_weights, up_lim, num_p, inp, order2_group=order2_group)
    elif method == 'arb-rc':
        optimal_split_list, mask_list = structural_searching_multip_alternating_group_rc(target_weights, up_lim, num_p, inp, order2_group=order2_group)
    elif method == 'braq':
        optimal_split_list, mask_list = structural_searching_multip(target_weights, up_lim, num_p, order2_group=order2_group)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list
