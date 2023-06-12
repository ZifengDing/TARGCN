import numpy as np
import torch
import time
import pickle
import math
import copy
from collections import defaultdict
import logging
import os
import sys

def setup_logger(name):
    cur_dir = os.getcwd()
    if not os.path.exists(cur_dir + '/log/'):
        os.mkdir(cur_dir + '/log/')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=cur_dir + '/log/' + name + '.log',
                        filemode='a')
    logger = logging.getLogger(name)
    return logger

def complex_mul(a, b):
    assert a.size(-1) == b.size(-1)
    dim = a.size(-1) // 2
    a_1, a_2 = torch.split(a, dim, dim=-1)
    b_1, b_2 = torch.split(b, dim, dim=-1)

    A = a_1 * b_1 - a_2 * b_2
    B = a_1 * b_2 + a_2 * b_1

    return torch.cat([A, B], dim=-1)

def get_dataset_stat(dataset):
    with open('dataset/' + dataset + '/stat.txt', 'r') as f:
        for line in f:
            line_ = line.strip().split()
            num_e, num_r, num_t = int(line_[0]), 2 * int(line_[1]), int(line_[2])
            break
    return num_e, num_r, num_t

class NeighborFinder:
    def __init__(self, adj, sampling=1, max_time=366 * 24, num_entities=None, weight_factor=1, time_granularity=24):
        self.time_granularity = time_granularity
        self.sampling = sampling
        self.weight_factor = weight_factor
        self.adj = adj

    def init_off_set(self, adj, max_time, num_entities):
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        off_set_t_l = []

        if isinstance(adj, list):
            for i in range(len(adj)):
                assert len(adj) == num_entities
                curr = adj[i]
                curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                curr_ts = [x[2] for x in curr]
                n_ts_l.extend(curr_ts)

                off_set_l.append(len(n_idx_l))
                off_set_t_l.append([np.searchsorted(curr_ts, cut_time, 'left') for cut_time in
                                    range(0, max_time + 1, self.time_granularity)])  # max_time+1 so we have max_time
        elif isinstance(adj, dict):
            for i in range(num_entities):
                curr = adj.get(i, [])
                curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                curr_ts = [x[2] for x in curr]
                n_ts_l.extend(curr_ts)

                off_set_l.append(len(n_idx_l))
                off_set_t_l.append([np.searchsorted(curr_ts, cut_time, 'left') for cut_time in
                                    range(0, max_time + 1, self.time_granularity)])  # max_time+1 so we have max_time

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l, off_set_t_l

    def set_adj(self, adj):
        self.adj = adj

    def get_temporal_neighbor(self, obj_idx_l, ts_l, num_neighbors=20):
        assert (len(obj_idx_l) == len(ts_l))

        out_ngh_node_batch = -np.ones((len(obj_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(obj_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_eidx_batch = -np.ones((len(obj_idx_l), num_neighbors)).astype(np.int32)
        offset_l = []
        got_node_emb_l = []

        if self.sampling == -1:
            full_ngh_node = []
            full_ngh_t = []
            full_ngh_edge = []
        for i, (obj_idx, cut_time) in enumerate(zip(obj_idx_l, ts_l)):
            if i == 0:
                been_through = 0  # a variable to track offset

            srt_l = self.adj[obj_idx]
            got_node_emb_l.append(0)
            if len(srt_l) == 0:
                offset_l.append([been_through, been_through])
                continue

            ngh_idx = np.array(srt_l).transpose()[0]
            ngh_eidx = np.array(srt_l).transpose()[1]
            ngh_ts = np.array(srt_l).transpose()[2]

            if len(ngh_idx) > 0:
                if self.sampling == 0:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    sampled_idx = np.sort(sampled_idx)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                elif self.sampling == 1:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                elif self.sampling == 2:
                    ngh_ts = ngh_ts[-num_neighbors:]
                    ngh_idx = ngh_idx[-num_neighbors:]
                    ngh_eidx = ngh_eidx[-num_neighbors:]
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                elif self.sampling == 3:
                    delta_t = (-abs(ngh_ts - cut_time)) / (self.time_granularity * self.weight_factor)
                    weights = np.exp(delta_t) + 1e-9
                    weights = weights / sum(weights)

                    if len(ngh_idx) >= num_neighbors:
                        sampled_idx = np.random.choice(len(ngh_idx), num_neighbors, replace=False, p=weights)
                    else:
                        sampled_idx = np.random.choice(len(ngh_idx), len(ngh_idx), replace=False, p=weights)

                    sampled_idx = np.sort(sampled_idx)
                    out_ngh_node_batch[i, num_neighbors - len(sampled_idx):] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, num_neighbors - len(sampled_idx):] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, num_neighbors - len(sampled_idx):] = ngh_eidx[sampled_idx]

                    if (num_neighbors - len(sampled_idx)) != 0:
                        offset_l.append([been_through, been_through + len(sampled_idx)])
                        been_through += len(sampled_idx)
                    else:
                        offset_l.append([been_through, been_through + num_neighbors])
                        been_through += num_neighbors

                elif self.sampling == 4:
                    weights = (ngh_ts + 1) / sum(ngh_ts + 1)

                    if len(ngh_idx) >= num_neighbors:
                        sampled_idx = np.random.choice(len(ngh_idx), num_neighbors, replace=False, p=weights)
                    else:
                        sampled_idx = np.random.choice(len(ngh_idx), len(ngh_idx), replace=False, p=weights)

                    sampled_idx = np.sort(sampled_idx)
                    out_ngh_node_batch[i, num_neighbors - len(sampled_idx):] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, num_neighbors - len(sampled_idx):] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, num_neighbors - len(sampled_idx):] = ngh_eidx[sampled_idx]

                elif self.sampling == -1:  # use whole neighborhood
                    full_ngh_node.append(ngh_idx[-200000:])
                    full_ngh_t.append(ngh_ts[-200000:])
                    full_ngh_edge.append(ngh_eidx[-200000:])
                else:
                    raise ValueError("invalid input for sampling")

        if self.sampling == -1:
            max_num_neighbors = max(map(len, full_ngh_edge))
            out_ngh_node_batch = -np.ones((len(obj_idx_l), max_num_neighbors)).astype(np.int32)
            out_ngh_t_batch = np.zeros((len(obj_idx_l), max_num_neighbors)).astype(np.int32)
            out_ngh_eidx_batch = -np.ones((len(obj_idx_l), max_num_neighbors)).astype(np.int32)
            for i in range(len(full_ngh_node)):
                out_ngh_node_batch[i, max_num_neighbors - len(full_ngh_node[i]):] = full_ngh_node[i]
                out_ngh_eidx_batch[i, max_num_neighbors - len(full_ngh_edge[i]):] = full_ngh_edge[i]
                out_ngh_t_batch[i, max_num_neighbors - len(full_ngh_t[i]):] = full_ngh_t[i]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, offset_l, got_node_emb_l