from re import L
import numpy as np
from pyparsing import line
import torch
from binary_arb import high_order_residual, high_order_residual_alternating_order1, high_order_residual_alternating_mean, high_order_residual_alternating_order2_rc_nomean, high_order_residual_alternating_order1_rc_nomean
from utils.mask import generate_structural_mask

error_N = 2048*4096*128

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

def error_computing_x_all_accelerate(origin_matrix, quantized_matrix, S):
    # inps shape [128, 2048, 128]
    R = (origin_matrix - quantized_matrix).T
    P = torch.einsum('ik,jk->ij', R, R)
    return torch.sum(P * S) / error_N

def calculate_percentage_and_variance_original(weights, abs_weights, bin_edges):
    percentages = []
    variances = []
    accum_percentages = [0]
    total_elements = abs_weights.numel()
    for i in range(len(bin_edges) - 1):
        bin_mask = (abs_weights >= bin_edges[i]) & (abs_weights < bin_edges[i + 1])
        bin_weights = weights[bin_mask]
        percentages.append(bin_weights.numel() / total_elements * 100)
        accum_percentages.append(accum_percentages[-1] + percentages[-1])
        variances.append(torch.var(bin_weights))
    return percentages, variances, accum_percentages

'''
Include main method to search the rate for 2-bit salient data columns and the optimal split for 1-bit data
'''
def structural_searching_multip(origin_matrix, up_lim=30, num_p=1, order2_group=False):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)
        group4 = high_order_residual(origin_matrix, ~mask3, order=2)


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_group(origin_matrix, up_lim=30, num_p=1, inp=None, iter=0, order2_group=False):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    # error = []
    # lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate
        group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate

        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        # error.append(quantize_error_0.item())
        # lines.append(i)
        # print(quantize_error_0)

        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True

    group3 = high_order_residual_alternating_mean(origin_matrix, mask3, order=2)

    mask_list = []
    optimal_split_list = []

    # 2nd order group
    if order2_group:
        mask0 = mask3.clone()
        minimal_value2 = float('inf')
        group0 = torch.zeros(origin_matrix.shape, device=origin_matrix.device)
        for i in range(num_p):
            search_matrix = origin_matrix * mask0

            flat_abs_tensor = torch.abs(search_matrix).view(-1)
            flat_abs_tensor_nonzero = flat_abs_tensor[flat_abs_tensor != 0]
            percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
            percentile_values = torch.tensor(
                np.quantile(flat_abs_tensor_nonzero.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
            ).to(origin_matrix.device)

            # search for the optimal split for the second group, high order=1,, non-structured search
            flag = False
            for split_value in percentile_values:
                mask4, mask5 = generate_structural_mask(origin_matrix, ~mask0, split_value)
                group1 = high_order_residual(origin_matrix, mask4, order=2)
                group2 = high_order_residual(origin_matrix, mask5, order=2)

                quantize_error = error_computing(origin_matrix, group1+group2+group0)
                if quantize_error < minimal_value2:
                    minimal_value2 = quantize_error
                    optimal_split = split_value
                    optimal_group2 = group2
                    best_mask4 = mask4
                    best_mask5 = mask5
                    flag = True

            if not flag:
                print(False, 2)
                optimal_split = percentile_values[0]
                best_mask4, best_mask5 = generate_structural_mask(origin_matrix, ~mask0, optimal_split)

            mask0 = mask0 & (~best_mask5)
            mask_list.append(best_mask5)
            group0 = group0 + optimal_group2

        mask_list.append(best_mask4)

    else:
        mask_list.append(mask3)

    # 1st order group
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        flag = False
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)

            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                best_mask2 = mask2
                best_mask1 = mask1
                flag = True

        if not flag:
            print(False)
            optimal_split = percentile_values[0]
            best_mask1, best_mask2 = generate_structural_mask(origin_matrix, mask3, optimal_split)
        
        optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)
        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_group_x(origin_matrix, up_lim=30, num_p=1, inp=None, iter=0, order2_group=False):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    # error = []
    # lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate
        group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate

        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        # error.append(quantize_error_0.item())
        # lines.append(i)
        # print(quantize_error_0)

        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True

    group3 = high_order_residual_alternating_mean(origin_matrix, mask3, order=2)

    mask_list = []
    optimal_split_list = []

    # 2nd order group
    if order2_group:
        mask0 = mask3.clone()
        minimal_value2 = float('inf')
        group0 = torch.zeros(origin_matrix.shape, device=origin_matrix.device)
        for i in range(num_p):
            search_matrix = origin_matrix * mask0

            flat_abs_tensor = torch.abs(search_matrix).view(-1)
            flat_abs_tensor_nonzero = flat_abs_tensor[flat_abs_tensor != 0]
            percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
            percentile_values = torch.tensor(
                np.quantile(flat_abs_tensor_nonzero.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
            ).to(origin_matrix.device)

            # search for the optimal split for the second group, high order=1,, non-structured search
            flag = False
            for split_value in percentile_values:
                mask4, mask5 = generate_structural_mask(origin_matrix, ~mask0, split_value)
                group1 = high_order_residual(origin_matrix, mask4, order=2)
                group2 = high_order_residual(origin_matrix, mask5, order=2)

                quantize_error = error_computing(origin_matrix, group1+group2+group0)
                if quantize_error < minimal_value2:
                    minimal_value2 = quantize_error
                    optimal_split = split_value
                    optimal_group2 = group2
                    best_mask4 = mask4
                    best_mask5 = mask5
                    flag = True

            if not flag:
                print(False, 2)
                optimal_split = percentile_values[0]
                best_mask4, best_mask5 = generate_structural_mask(origin_matrix, ~mask0, optimal_split)

            mask0 = mask0 & (~best_mask5)
            mask_list.append(best_mask5)
            group0 = group0 + optimal_group2

        mask_list.append(best_mask4)

    else:
        mask_list.append(mask3)

    # 1st order group
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        flag = False
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)

            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing_x_all_accelerate(origin_matrix, group1+group2+group3, inp)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                best_mask2 = mask2
                best_mask1 = mask1
                flag = True

        if not flag:
            print(False)
            optimal_split = percentile_values[0]
            best_mask1, best_mask2 = generate_structural_mask(origin_matrix, mask3, optimal_split)
        
        optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)   # accelerate
        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_group_rc(origin_matrix, up_lim=30, num_p=1, inp=None, iter=0, order2_group=False):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    # error = []
    # lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate
        group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate

        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        # error.append(quantize_error_0.item())
        # lines.append(i)
        # print(quantize_error_0)

        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True

    group3 = high_order_residual_alternating_order2_rc_nomean(origin_matrix, mask3, order=2)

    mask_list = []
    optimal_split_list = []

    # 2nd order group
    if order2_group:
        mask0 = mask3.clone()
        minimal_value2 = float('inf')
        group0 = torch.zeros(origin_matrix.shape, device=origin_matrix.device)
        for i in range(num_p):
            search_matrix = origin_matrix * mask0

            flat_abs_tensor = torch.abs(search_matrix).view(-1)
            flat_abs_tensor_nonzero = flat_abs_tensor[flat_abs_tensor != 0]
            percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
            percentile_values = torch.tensor(
                np.quantile(flat_abs_tensor_nonzero.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
            ).to(origin_matrix.device)

            # search for the optimal split for the second group, high order=1,, non-structured search
            flag = False
            for split_value in percentile_values:
                mask4, mask5 = generate_structural_mask(origin_matrix, ~mask0, split_value)
                group1 = high_order_residual(origin_matrix, mask4, order=2)
                group2 = high_order_residual(origin_matrix, mask5, order=2)

                quantize_error = error_computing(origin_matrix, group1+group2+group0)
                if quantize_error < minimal_value2:
                    minimal_value2 = quantize_error
                    optimal_split = split_value
                    optimal_group2 = group2
                    best_mask4 = mask4
                    best_mask5 = mask5
                    flag = True

            if not flag:
                print(False, 2)
                optimal_split = percentile_values[0]
                best_mask4, best_mask5 = generate_structural_mask(origin_matrix, ~mask0, optimal_split)

            mask0 = mask0 & (~best_mask5)
            mask_list.append(best_mask5)
            group0 = group0 + optimal_group2

        mask_list.append(best_mask4)

    else:
        mask_list.append(mask3)

    # 1st order group
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        flag = False
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)

            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            # quantize_error = error_computing_x_all_accelerate(origin_matrix, group1+group2+group3, inp)
            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                best_mask2 = mask2
                best_mask1 = mask1
                flag = True

        if not flag:
            print(False)
            optimal_split = percentile_values[0]
            best_mask1, best_mask2 = generate_structural_mask(origin_matrix, mask3, optimal_split)
        
        # optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)
        optimal_group2 = high_order_residual_alternating_order1_rc_nomean(origin_matrix, best_mask2, order=1, iter=0)
        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def find_optimal_split(group_max, origin_matrix, border):
    optimal_split = None
    minimal_value = float('inf')
    searching_steps = torch.arange(0.1,0.8,0.01)
    searching_steps = searching_steps * group_max

    group3 = high_order_residual(origin_matrix, torch.abs(origin_matrix) > border, order=2)
    for split_value in searching_steps:

        group1 = high_order_residual(origin_matrix, (torch.abs(origin_matrix) > split_value) & (torch.abs(origin_matrix) <= border), order=1)
        group2 = high_order_residual(origin_matrix, torch.abs(origin_matrix) <= split_value, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value

    return optimal_split, minimal_value
