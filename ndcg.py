import os
from tqdm import tqdm
import numpy as np
import sys
import torch
from sklearn import metrics


class AverageNDCGMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, ndcg_at_k=[1, 3, 5, 10, 20, 100000000]):
        self.ndcg = {}
        self.ndcg_at_k = ndcg_at_k
        
        self.reset()

    def reset(self):
        for k in self.ndcg_at_k:
            self.ndcg[k] = []

    def value(self):
        for k in self.ndcg:
            self.ndcg[k] = torch.mean(torch.stack(self.ndcg[k]))

        return self.ndcg


    def compute_dcg_at_k(self, relevances, k):
        dcg = 0
        for i in torch.arange(min(len(relevances), k)):
            dcg += (2 ** relevances[i] - 1) / torch.log2(i + 2)  # +2 as we start our idx at 0
        return dcg


    def compute_ndcg_at_k(self, predicted_relevance, true_relevances):
        # NDCG@k
        for k_val in self.ndcg_at_k:
            predicted = self.compute_dcg_at_k(predicted_relevance, k_val)
            true = self.compute_dcg_at_k(true_relevances, k_val)
            if true <= 1e-6:
                ndcg_value = 1
            else:
                ndcg_value = predicted / true  
            self.ndcg[k_val].append(ndcg_value)

    def compute_ndcg_at_k_batch(self, predicted_relevance, true_relevances):
        # NDCG@k
        assert predicted_relevance.shape == true_relevances.shape
        for i in range(predicted_relevance.shape[0]):
            for k_val in self.ndcg_at_k:
                ndcg_value = self.compute_dcg_at_k(predicted_relevance[i], k_val) / self.compute_dcg_at_k(true_relevances[i], k_val)
                self.ndcg[k_val].append(ndcg_value)
    
    def return_ndcg_at_k(self, predicted_relevance, true_relevances):
        # NDCG@k
        ndcg_value_list = []
        for k_val in self.ndcg_at_k:
            predicted = self.compute_dcg_at_k(predicted_relevance, k_val)
            true = self.compute_dcg_at_k(true_relevances, k_val)
            if true <= 1e-6:
                ndcg_value = torch.ones(1)[0].cuda()
            else:
                ndcg_value = predicted / true  
            ndcg_value_list.append(ndcg_value)
        return torch.stack(ndcg_value_list)