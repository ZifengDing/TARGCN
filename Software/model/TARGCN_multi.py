import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging
import numpy as np
import copy


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific

        if entity_specific:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
                    num_entities, 1))
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_entities, 1))
        else:
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  # shape: num_entities * time_dim
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities=None):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = torch.unsqueeze(ts, dim=2)
        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(
                dim=1)  # self.basis_freq[entities]:  [batch_size, time_dim]
            map_ts += self.phase[entities].unsqueeze(dim=1)
        else:
            map_ts = ts * self.basis_freq.view(1, 1, -1)  # [batch_size, 1, time_dim]
            map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class TARGCN(nn.Module):
    def __init__(self, neighbor_finder, embed_dim, num_ent, num_rel, logger, decoder, steps=2, device='cpu'):
        super(TARGCN, self).__init__()
        self.device = device
        self.nf = neighbor_finder
        self.embed_dim = embed_dim
        self.pad_idx = num_ent + num_rel
        num_symbols = num_ent + num_rel
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.decoder = decoder
        self.num_symbols = num_symbols
        self.logger = logger
        self.steps = steps

        self.use_time_embedding = True
        self.loss_func = torch.nn.CrossEntropyLoss()

        if decoder.lower() == 'distmult':
            self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols).to(self.device)

            self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
            self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
            if self.use_time_embedding:
                self.node_emb_proj = nn.Linear(2 * embed_dim, embed_dim)
                self.time_encoder = TimeEncode(expand_dim=embed_dim, entity_specific=False,
                                               num_entities=num_ent, device=self.device)
            else:
                self.node_emb_proj = nn.Linear(embed_dim, embed_dim)

        elif decoder.lower() == 'bique':
            self.rank = 25
            init_size = 1e-3
            self.ent_emb_bi = nn.Embedding(self.num_ent, 8 * self.rank, sparse=False).to(self.device)
            self.rel_emb_bi = nn.Embedding(self.num_rel, 16 * self.rank, sparse=False).to(self.device)
            self.ent_emb_bi.weight.data *= init_size
            self.rel_emb_bi.weight.data *= init_size

            self.gcn_w = nn.Linear(8 * self.rank + 16 * self.rank, 8 * self.rank)
            self.gcn_b = nn.Parameter(torch.FloatTensor(8 * self.rank))

            if self.use_time_embedding:  # for BiQUE, time embedding dimension is still equal to embsize in hyperparameter
                self.node_emb_proj = nn.Linear(8 * self.rank + embed_dim, 8 * self.rank)
                self.time_encoder = TimeEncode(expand_dim=embed_dim, entity_specific=False,
                                               num_entities=num_ent, device=self.device)
            else:
                self.node_emb_proj = nn.Linear(8 * self.rank, 8 * self.rank)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

    def find_temporal_neighbor(self, search_idx_all, ts_idx_all, offset_all, got_node_emb_all, neighbors,
                               num_neighbors):
        search_idx_l = search_idx_all[-1]
        ts_idx_l = ts_idx_all[-1]

        out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, offset_l, got_node_emb_l = \
            self.nf.get_temporal_neighbor(search_idx_l, ts_idx_l, num_neighbors=num_neighbors)

        mask = out_ngh_node_batch.flatten() != -1
        neighbors.append([out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch])
        search_idx_all.append(out_ngh_node_batch.flatten()[mask])
        ts_idx_all.append(out_ngh_t_batch.flatten()[mask])
        offset_all.append(offset_l)
        got_node_emb_all.append(got_node_emb_l)

        return neighbors, search_idx_all, ts_idx_all, offset_all, got_node_emb_all

    def get_ent_emb(self, ent_idx_l):
        if self.decoder.lower() == 'distmult':
            return self.symbol_emb(torch.from_numpy(ent_idx_l).long().to(self.device))
        elif self.decoder.lower() == 'bique':
            return self.ent_emb_bi(torch.from_numpy(ent_idx_l).long().to(self.device))

    def get_rel_emb(self, rel_idx_l):
        if self.decoder.lower() == 'distmult':
            return self.symbol_emb(torch.from_numpy(rel_idx_l + self.num_ent).long().to(self.device))
        elif self.decoder.lower() == 'bique':
            return self.rel_emb_bi(torch.from_numpy(rel_idx_l).long().to(self.device))

    def get_node_emb(self, ngh_idx_l, ngh_time_l, former_ts_idx):
        hidden_node = self.get_ent_emb(ngh_idx_l)
        if self.use_time_embedding:
            cut_time_l = ngh_time_l - former_ts_idx
            hidden_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device))
            return self.node_emb_proj(torch.cat([hidden_node, torch.squeeze(hidden_time, 1)], axis=1))
        else:
            return self.node_emb_proj(hidden_node)

    def neighbor_encoder(self, node_emb, rel_emb):
        concat_emb = torch.cat((node_emb, rel_emb), dim=-1)
        out = self.gcn_w(concat_emb)
        out = torch.sum(out, dim=0)
        out = out / node_emb.shape[0]
        return out.tanh()

    def forward(self, batch, num_neighbors=20):
        search_idx_all = [batch.src_idx]
        rel_idx_all = [batch.rel_idx]
        ts_idx_all = [batch.ts]
        offset_all = []
        got_node_emb_all = []  # whether we have got the node embedding before aggregation
        neighbors = []

        # find multi-hop temporal neighborhood (backward, expand graph)
        for step in range(self.steps):
            neighbors, search_idx_all, ts_idx_all, offset_all, got_node_emb_all = \
                self.find_temporal_neighbor(search_idx_all, ts_idx_all, offset_all, got_node_emb_all, neighbors,
                                            num_neighbors)

        # initialize node emb of the latest found neighbors
        neighbors_latest = neighbors[-1]
        ngh_emb, ngh_rel_emb = [], []
        ngh_emb_cur, ngh_rel_emb_cur = [], []
        for i, ngh_l in enumerate(neighbors_latest[0]):
            mask = (neighbors_latest[0][i] != -1)  # filter out the '-1' terms when we cannot sample same number of neighors as num_neighbors
            ngh_emb_cur.append(self.get_node_emb(neighbors_latest[0][i][mask], neighbors_latest[2][i][mask],
                                                 [ts_idx_all[-2][i]] * len(neighbors_latest[0][i][mask])))  # get node embedding (ent, ts)
            ngh_rel_emb_cur.append(self.get_rel_emb(neighbors_latest[1][i][mask]))  # get relation embedding of the nodes
            got_node_emb_all[-1][i] = 1
            if True not in mask:
                got_node_emb_all[-1][i] = 0  # do not get node embedding since we could not find neighbors before

        ngh_emb.append(ngh_emb_cur)  # collect the embeddings from the latest found neighbors
        ngh_rel_emb.append(ngh_rel_emb_cur)
        ngh_emb_cur, ngh_rel_emb_cur, ngh_emb_cur_, sub_emb_all = [], [], [], []

        # aggregate from the latest found neighbors (forward, embedding aggregation)
        for step in range(self.steps - 1):
            offset_cur_ = copy.deepcopy(offset_all[-1 - step - 1])
            j = 0  # a pointer to take time of prior nodes
            for i, node in enumerate(search_idx_all[self.steps - step - 1]):
                ngh_emb_, ngh_rel_emb_ = ngh_emb[step][i], ngh_rel_emb[step][i]
                if i >= offset_cur_[0][1]:
                    offset_cur_ = offset_cur_[1:]
                    j += 1
                if ngh_emb_.shape[0] == 0:
                    t_prior = ts_idx_all[self.steps - step - 2][j]
                    aggregated_emb = self.get_node_emb(np.array(search_idx_all[self.steps - step - 1][i]).astype(np.int32),
                                                       np.array(ts_idx_all[self.steps - step - 1][i]).astype(np.int32),
                                                       np.array(t_prior).astype(np.int32))
                else:
                    aggregated_emb = self.neighbor_encoder(ngh_emb_, ngh_rel_emb_)
                ngh_emb_cur_.append(aggregated_emb)  # the aggregated embedding is stored
            offset_cur = offset_all[-1 - step - 1]  # offset of current step
            ngh_emb_cur = [ngh_emb_cur_[k[0]:k[1]] for k in offset_cur]
            ngh_emb_cur = [(torch.stack(node_embedding_cur, dim=0) if len(node_embedding_cur) > 0 else self.get_ent_emb(np.array([]).astype(np.int32)))
                           for node_embedding_cur in ngh_emb_cur]

            for i in range(len(neighbors[-1 - step - 1][0])):  # for every node in this layer of neighbors
                mask = (neighbors[-1 - step - 1][0][i] != -1)  # filter out the '-1' terms when we cannot sample same number of neighors as num_neighbors
                ngh_rel_emb_cur.append(self.get_rel_emb(neighbors[-1 - step - 1][1][i][mask]))  # get relation embedding of the nodes

            ngh_emb.append(ngh_emb_cur)
            ngh_rel_emb.append(ngh_rel_emb_cur)
            ngh_emb_cur, ngh_rel_emb_cur, ngh_emb_cur_ = [], [], []

        for i in range(batch.src_idx.shape[0]):
            ngh_emb_, ngh_rel_emb_ = ngh_emb[-1][i], ngh_rel_emb[-1][i]
            if ngh_emb_.shape[0] == 0:  # have not get node embedding yet
                sub_emb = self.get_node_emb(np.array([batch.src_idx[i]]).astype(np.int32),
                                            np.zeros((1,)).astype(np.int32), np.zeros((1,)).astype(np.int32))
                sub_emb = torch.sum(sub_emb, dim=0)
                sub_emb = sub_emb.tanh()
            else:
                sub_emb = self.neighbor_encoder(ngh_emb_, ngh_rel_emb_)

            sub_emb_all.append(sub_emb)

        sub_emb_all = torch.stack(sub_emb_all, dim=0)

        if self.decoder.lower() == 'distmult':
            score = self.Distmult(sub_emb_all, batch)
        elif self.decoder.lower() == 'bique':
            score = self.BiQUE(sub_emb_all, batch)

        return score

    def Distmult(self, sub_emb, batch):
        all_emb = self.get_node_emb(np.arange(self.num_ent), np.zeros((self.num_ent,)).astype(np.int32),
                                    np.zeros((self.num_ent,)).astype(np.int32))  # set time difference equal to 0
        rel_emb = self.get_rel_emb(batch.rel_idx)

        match_emb = sub_emb * rel_emb
        all_score = torch.mm(match_emb, all_emb.transpose(1, 0))

        return all_score

    def BiQUE(self, sub_emb, batch):
        all_emb = self.get_node_emb(np.arange(self.num_ent), np.zeros((self.num_ent,)).astype(np.int32),
                                    np.zeros((self.num_ent,)).astype(np.int32))  # set time difference equal to 0
        rel_emb = self.get_rel_emb(batch.rel_idx)

        sub_emb += rel_emb[:, self.rank * 8:]
        w_a, x_a, y_a, z_a = torch.split(sub_emb, self.rank * 2, dim=-1)
        w_b, x_b, y_b, z_b = torch.split(rel_emb[:, :self.rank * 8], self.rank * 2, dim=-1)

        A = self.complex_mul(w_a, w_b) - self.complex_mul(x_a, x_b) - self.complex_mul(y_a, y_b) - self.complex_mul(z_a,
                                                                                                                    z_b)
        B = self.complex_mul(w_a, x_b) + self.complex_mul(x_a, w_b) + self.complex_mul(y_a, z_b) - self.complex_mul(z_a,
                                                                                                                    y_b)
        C = self.complex_mul(w_a, y_b) - self.complex_mul(x_a, z_b) + self.complex_mul(y_a, w_b) + self.complex_mul(z_a,
                                                                                                                    x_b)
        D = self.complex_mul(w_a, z_b) + self.complex_mul(x_a, y_b) - self.complex_mul(y_a, x_b) + self.complex_mul(z_a,
                                                                                                                    w_b)

        match_emb = torch.cat([A, B, C, D], dim=-1)
        all_score = torch.mm(match_emb, all_emb.transpose(1, 0))
        return all_score

    def loss(self, score, obj):
        return self.loss_func(score, obj)

    def complex_mul(self, a, b):
        assert a.size(-1) == b.size(-1)
        dim = a.size(-1) // 2
        a_1, a_2 = torch.split(a, dim, dim=-1)
        b_1, b_2 = torch.split(b, dim, dim=-1)

        A = a_1 * b_1 - a_2 * b_2
        B = a_1 * b_2 + a_2 * b_1

        return torch.cat([A, B], dim=-1)