import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

class Preprocessor():
    def __init__(self, num_pipes: int, num_nodes: int):

        self.num_pipes = num_pipes
        self.num_nodes = num_nodes

        self.min_flow_rate = 1e8*torch.ones(num_pipes)
        self.max_flow_rate = -1e8*torch.ones(num_pipes)

        self.min_head = 1e8*torch.ones(num_nodes)
        self.max_head = -1e8*torch.ones(num_nodes)

    def partial_fit(self, state):
        flow_rate = state[:, 0:self.num_pipes]
        head = state[:, self.num_pipes:self.num_pipes+self.num_nodes]

        '''
        if torch.any(flow_rate < self.min_flow_rate):
            self.min_flow_rate = torch.min(flow_rate)
        if torch.any(flow_rate > self.max_flow_rate):
            self.max_flow_rate = torch.max(flow_rate, dim=0)[0]

        if torch.any(head < self.min_head):
            self.min_head = torch.min(head)
        if torch.any(head > self.max_head):
            self.max_head = torch.max(head)

        '''
        if torch.any(torch.any(flow_rate < self.min_flow_rate, dim=0)).item():
            ids = torch.where(torch.any(flow_rate < self.min_flow_rate, dim=0))[0]
            self.min_flow_rate[ids] = torch.min(flow_rate[:, ids], dim=0)[0]
        if torch.any(torch.any(flow_rate > self.max_flow_rate, dim=0)).item():
            ids = torch.where(torch.any(flow_rate > self.max_flow_rate, dim=0))[0]
            self.max_flow_rate[ids] = torch.max(flow_rate[:, ids], dim=0)[0]

        if torch.any(torch.any(head < self.min_head, dim=0)).item():
            ids = torch.where(torch.any(head < self.min_head, dim=0))[0]
            self.min_head[ids] = torch.min(head[:, ids], dim=0)[0]
        if torch.any(torch.any(head > self.max_head, dim=0)).item():
            ids = torch.where(torch.any(head > self.max_head, dim=0))[0]
            self.max_head[ids] = torch.max(head[:, ids], dim=0)[0]

        self.flow_rate_min_equal_max_ids = \
            torch.where(torch.abs(self.max_flow_rate - self.min_flow_rate) < 1e-12)[0]
        self.head_min_equal_max_ids = \
            torch.where(torch.abs(self.max_head - self.min_head) < 1e-12)[0]

        self.flow_rate_transform_ids = \
            torch.where(torch.abs(self.max_flow_rate - self.min_flow_rate) > 1e-12)[0]
        self.head_transform_ids = \
            torch.where(torch.abs(self.max_head - self.min_head) > 1e-12)[0]

    def transform_state(self, state):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        flow_rate = state[:, 0:self.num_pipes]
        head = state[:, self.num_pipes:self.num_pipes+self.num_nodes]

        '''
        flow_rate = (flow_rate - self.min_flow_rate) / (self.max_flow_rate - self.min_flow_rate)
        head = (head - self.min_head) / (self.max_head - self.min_head)

        '''
        flow_rate[:, self.flow_rate_min_equal_max_ids] = 0.5
        head[:, self.head_min_equal_max_ids] = 0.5

        flow_rate[:, self.flow_rate_transform_ids] = \
            (flow_rate[:, self.flow_rate_transform_ids] - self.min_flow_rate[self.flow_rate_transform_ids]) \
            / (self.max_flow_rate[self.flow_rate_transform_ids] - self.min_flow_rate[self.flow_rate_transform_ids])
        head[:, self.head_transform_ids] = \
            (head[:, self.head_transform_ids] - self.min_head[self.head_transform_ids]) \
            / (self.max_head[self.head_transform_ids] - self.min_head[self.head_transform_ids])

        state = torch.cat((flow_rate, head), dim=1).squeeze(0)
        return state

    def inverse_transform_states(self, states):
        flow_rate = states[:, 0:self.num_pipes]
        head = states[:, self.num_pipes:self.num_pipes+self.num_nodes]

        flow_rate = flow_rate * (self.max_flow_rate - self.min_flow_rate) + self.min_flow_rate
        head = head * (self.max_head - self.min_head) + self.min_head

        state = torch.cat((flow_rate, head), dim=1).squeeze(0)
        return state

