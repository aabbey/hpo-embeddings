import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from scipy.stats import ks_2samp

from hpo_extract.compare_phenos.eval_helpers import evaluate_matrix


class PhenoConnect6(nn.Module):
    def __init__(self, g_len, h_len, hidden_len=4):
        super(PhenoConnect6, self).__init__()
        self.g_len = g_len
        self.hpo_par = nn.Parameter(
            torch.normal(mean=0.1, std=0.2, size=(1, 1, hidden_len, h_len))
        )
        self.pow = nn.Parameter(torch.normal(mean=-2.5, std=1, size=(1, hidden_len)))
        self.bias = nn.Parameter(torch.normal(mean=-3.0, std=1, size=(1, hidden_len)))
        self.genes_hpo_scale = nn.Parameter(
            torch.normal(mean=0.5, std=0.1, size=(1, 1, hidden_len, 1))
        )
        self.hpo_bias = nn.Parameter(
            torch.normal(mean=-1.0, std=0.2, size=(1, 1, hidden_len, 1))
        )
        self.gen_bias = nn.Parameter(
            torch.normal(mean=-3.0, std=1, size=(1, 1, hidden_len, 1))
        )

        self.linear1 = nn.Linear(hidden_len, hidden_len * 2)
        self.linear2 = nn.Linear(hidden_len * 2, 1)

    def forward(self, gh_sum, h_sum, g_sum):
        gh_sum, h_sum, g_sum = (
            gh_sum.unsqueeze(2),
            h_sum.unsqueeze(2),
            g_sum.unsqueeze(2),
        )  # (b, 2, hd, h) (b, 2, hd, h) (b, 2, hd, 1)
        x2 = (
            gh_sum * self.genes_hpo_scale
            + h_sum * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + g_sum * self.genes_hpo_scale * torch.exp(self.hpo_bias)
            + self.g_len
            * (
                torch.exp(self.gen_bias)
                * torch.exp(self.hpo_bias)
                * self.genes_hpo_scale
            )
        )  # (b, 2, hd, h)

        x4 = x2 * self.hpo_par
        x5 = torch.sum(torch.prod(x4, dim=1), dim=-1)
        combined_power = x5 ** torch.exp(self.pow)
        combined_power2 = combined_power + self.bias
        sig = torch.sigmoid(combined_power2)
        out1 = torch.sigmoid(self.linear1(sig))
        return torch.sigmoid(self.linear2(out1))


class PhenoConnect7(nn.Module):
    def __init__(self, g_len, h_len, hidden_len, dropout=0.1):
        super(PhenoConnect7, self).__init__()
        self.hidden_len = hidden_len
        self.g_len = g_len
        self.hpo_par = nn.Parameter(
            torch.normal(mean=0.1, std=0.02, size=(hidden_len, 1, h_len))
        )
        # self.pow = nn.Parameter(torch.normal(mean=-2.5, std=1, size=((1, hidden_len))))
        self.out_scale = nn.Parameter(
            torch.normal(mean=0.001, std=0.0005, size=((1, hidden_len)))
        )
        self.bias = nn.Parameter(torch.normal(mean=-50.0, std=10, size=(1, hidden_len)))
        self.genes_hpo_scale = nn.Parameter(
            torch.normal(mean=0.5, std=0.1, size=(1, hidden_len, 1))
        )
        self.hpo_bias = nn.Parameter(
            torch.normal(mean=-1.0, std=0.2, size=(1, hidden_len, 1))
        )
        self.gen_bias = nn.Parameter(
            torch.normal(mean=-3.0, std=1, size=(1, hidden_len, 1))
        )

        self.linear1 = nn.Linear(hidden_len, hidden_len * 2)
        self.act1 = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_len * 2, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, gh1, h1, gh2, h2):
        # (b, hl) (b, hl) (b, hl) (b, hl)
        gh1, h1, gh2, h2 = (
            gh1.unsqueeze(1),
            h1.unsqueeze(1),
            gh2.unsqueeze(1),
            h2.unsqueeze(1),
        )

        # print(gh1[0][gh1[0] != 0].mean())

        x1 = (
            gh1 * self.genes_hpo_scale
            + h1 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -2
        )  # (b, hd, 1, h)
        # print(x1[0, 0, 0])

        x2 = (
            gh2 * self.genes_hpo_scale
            + h2 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)
        # print(x2[0, 0, :, 0])

        hpar = self.dropout(self.hpo_par)
        x1, x2 = x1 * hpar, x2 * hpar.transpose(-2, -1)

        x3 = x1 @ x2  # (b, hd, 1, 1)
        # print("out x3 : ", x3.squeeze()[0, 0])
        # out = x3.squeeze() ** torch.exp(self.pow)  # (b, hd)
        out = x3.squeeze() ** 0.5
        # print("out1 : ", out)
        out = out * self.out_scale + self.bias
        # print("out scale and bias 0 : ", out[:, 0])
        # out = torch.sigmoid(out)
        # print("out sigmoid : ", out)
        # print("out linear : ", self.linear1(out))
        out = self.act1(self.linear1(out))
        # print("out leaky 0 : ", out[:, 0])
        # print("out linear2 : ", self.linear2(out))
        return torch.sigmoid(self.linear2(out))  # (b, hd)


class PhenoConnect7CS(nn.Module):
    def __init__(self, g_len, h_len, hidden_len, dropout=0.1):
        super(PhenoConnect7CS, self).__init__()
        self.hidden_len = hidden_len
        self.g_len = g_len
        self.hpo_par = nn.Parameter(
            torch.normal(mean=0.1, std=0.02, size=(hidden_len, h_len, 1))
        )
        self.out_scale = nn.Parameter(
            torch.normal(mean=1.0, std=0.1, size=((1, hidden_len)))
        )
        self.bias = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(1, hidden_len)))
        self.genes_hpo_scale = nn.Parameter(
            torch.normal(mean=0.5, std=0.1, size=(1, hidden_len, 1))
        )
        self.hpo_bias = nn.Parameter(
            torch.normal(mean=-1.0, std=0.2, size=(1, hidden_len, 1))
        )
        self.gen_bias = nn.Parameter(
            torch.normal(mean=-3.0, std=1, size=(1, hidden_len, 1))
        )

        self.linear1 = nn.Linear(hidden_len, hidden_len * 2)
        self.act1 = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_len * 2, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, gh1, h1, gh2, h2):
        # (b, hl) (b, hl) (b, hl) (b, hl)
        gh1, h1, gh2, h2 = (
            gh1.unsqueeze(1),
            h1.unsqueeze(1),
            gh2.unsqueeze(1),
            h2.unsqueeze(1),
        )

        # print(gh1[0][gh1[0] != 0].mean())

        x1 = (
            gh1 * self.genes_hpo_scale
            + h1 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)
        # print(x1[0, 0, 0])

        x2 = (
            gh2 * self.genes_hpo_scale
            + h2 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)
        # print(x2[0, 0, :, 0])

        hpar = self.dropout(self.hpo_par)
        x1, x2 = x1 * hpar, x2 * hpar

        x_cos = self.cos_sim(x1, x2)

        out = x_cos.squeeze() * self.out_scale + self.bias

        out = self.act1(self.linear1(out))

        return torch.sigmoid(self.linear2(out))  # (b, hd)


class PhenoConnect7CSPP(nn.Module):
    def __init__(self, g_len, h_len, hidden_len, dropout=0.1):
        super(PhenoConnect7CSPP, self).__init__()
        self.hidden_len = hidden_len
        self.g_len = g_len
        self.hpo_par = nn.Parameter(
            torch.normal(mean=0.1, std=0.02, size=(hidden_len, h_len, 1))
        )
        self.out_scale = nn.Parameter(
            torch.normal(mean=1.0, std=0.1, size=((1, hidden_len)))
        )
        self.bias = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(1, hidden_len)))
        self.genes_hpo_scale = nn.Parameter(
            torch.normal(mean=0.5, std=0.1, size=(1, hidden_len, 1))
        )
        self.hpo_bias = nn.Parameter(
            torch.normal(mean=-1.0, std=0.2, size=(1, hidden_len, 1))
        )
        self.gen_bias = nn.Parameter(
            torch.normal(mean=-3.0, std=1, size=(1, hidden_len, 1))
        )

        self.linear1 = nn.Linear(hidden_len + 1, hidden_len * 2)  # one extra for pp
        self.act1 = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_len * 2, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, gh1, h1, gh2, h2, phenopy_res):
        # (b, hl) (b, hl) (b, hl) (b, hl)
        gh1, h1, gh2, h2 = (
            gh1.unsqueeze(1),
            h1.unsqueeze(1),
            gh2.unsqueeze(1),
            h2.unsqueeze(1),
        )

        # print(gh1[0][gh1[0] != 0].mean())

        x1 = (
            gh1 * self.genes_hpo_scale
            + h1 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)
        # print(x1[0, 0, 0])

        x2 = (
            gh2 * self.genes_hpo_scale
            + h2 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)
        # print(x2[0, 0, :, 0])

        hpar = self.dropout(self.hpo_par)
        x1, x2 = x1 * hpar, x2 * hpar

        x_cos = self.cos_sim(x1, x2)

        out = x_cos.squeeze() * self.out_scale + self.bias

        out = torch.cat((out, phenopy_res), dim=1)

        out = self.act1(self.linear1(out))

        return torch.sigmoid(self.linear2(out))  # (b, hd)


class PhenoConnectPP(nn.Module):
    def __init__(self, g_len, h_len, hidden_len, dropout=0.1):
        super(PhenoConnectPP, self).__init__()
        self.hidden_len = hidden_len
        self.g_len = g_len
        self.hpo_par = nn.Parameter(
            torch.normal(mean=0.1, std=0.02, size=(hidden_len, 1, h_len))
        )
        self.out_scale = nn.Parameter(
            torch.normal(mean=0.001, std=0.0005, size=((1, hidden_len)))
        )
        self.bias = nn.Parameter(torch.normal(mean=-50.0, std=10, size=(1, hidden_len)))
        self.genes_hpo_scale = nn.Parameter(
            torch.normal(mean=0.5, std=0.1, size=(1, hidden_len, 1))
        )
        self.hpo_bias = nn.Parameter(
            torch.normal(mean=-1.0, std=0.2, size=(1, hidden_len, 1))
        )
        self.gen_bias = nn.Parameter(
            torch.normal(mean=-3.0, std=1, size=(1, hidden_len, 1))
        )

        self.linear1 = nn.Linear(hidden_len + 1, hidden_len * 2)  # one extra for pp
        self.act1 = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_len * 2, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, gh1, h1, gh2, h2, phenopy_res):
        # (b, hl) (b, hl) (b, hl) (b, hl)
        gh1, h1, gh2, h2 = (
            gh1.unsqueeze(1),
            h1.unsqueeze(1),
            gh2.unsqueeze(1),
            h2.unsqueeze(1),
        )

        x1 = (
            gh1 * self.genes_hpo_scale
            + h1 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -2
        )  # (b, hd, 1, h)

        x2 = (
            gh2 * self.genes_hpo_scale
            + h2 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)

        hpar = self.dropout(self.hpo_par)
        x1, x2 = x1 * hpar, x2 * hpar.transpose(-2, -1)

        x3 = x1 @ x2  # (b, hd, 1, 1)
        out = x3.squeeze() ** 0.5
        out = out * self.out_scale + self.bias

        out = torch.cat((out, phenopy_res), dim=1)  # (b, hd+1)

        out = self.act1(self.linear1(out))

        return torch.sigmoid(self.linear2(out))  # (b, 1)


class PhenoConnect7CSPP_logits(nn.Module):
    def __init__(self, g_len, h_len, hidden_len, dropout1=0.1, dropout2=0.1, pp=False):
        super(PhenoConnect7CSPP_logits, self).__init__()
        self.hidden_len = hidden_len
        self.g_len = g_len
        self.hpo_par = nn.Parameter(
            torch.normal(mean=0.1, std=0.02, size=(hidden_len, h_len, 1))
        )
        self.out_scale = nn.Parameter(
            torch.normal(mean=1.3, std=0.3, size=((1, hidden_len)))
        )
        self.bias = nn.Parameter(torch.normal(mean=0.1, std=0.1, size=(1, hidden_len)))
        self.genes_hpo_scale = nn.Parameter(
            torch.normal(mean=0.4, std=0.1, size=(1, hidden_len, 1))
        )
        self.hpo_bias = nn.Parameter(
            torch.normal(mean=-1.0, std=0.2, size=(1, hidden_len, 1))
        )
        self.gen_bias = nn.Parameter(
            torch.normal(mean=-3.0, std=1, size=(1, hidden_len, 1))
        )

        if pp:
            self.linear1 = nn.Linear(hidden_len + 1, hidden_len * 2)  # one extra for pp
        else:
            self.linear1 = nn.Linear(hidden_len, hidden_len * 2)
        self.act1 = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_len * 2, 1)

        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, gh1, h1, gh2, h2, phenopy_res=None):
        # (b, hl) (b, hl) (b, hl) (b, hl)
        gh1, h1, gh2, h2 = (
            gh1.unsqueeze(1),
            h1.unsqueeze(1),
            gh2.unsqueeze(1),
            h2.unsqueeze(1),
        )

        # print(gh1[0][gh1[0] != 0].mean())

        x1 = (
            gh1 * self.genes_hpo_scale
            + h1 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)
        # print(x1[0, 0, 0])

        x2 = (
            gh2 * self.genes_hpo_scale
            + h2 * self.genes_hpo_scale * torch.exp(self.gen_bias)
            + self.g_len
            * self.genes_hpo_scale
            * torch.exp(self.hpo_bias)
            * (1 + torch.exp(self.gen_bias))
        ).unsqueeze(
            -1
        )  # (b, hd, h, 1)
        # print(x2[0, 0, :, 0])

        hpar = self.dropout1(self.hpo_par)
        x1, x2 = x1 * hpar, x2 * hpar

        x_cos = self.cos_sim(x1, x2)

        out = x_cos.squeeze() * self.out_scale + self.bias

        out = (
            torch.cat((out, phenopy_res.unsqueeze(1)), dim=1)
            if phenopy_res is not None
            else out
        )

        out = self.act1(self.linear1(out))
        out = self.dropout2(out)
        return self.linear2(out)  # (b, 1)


class CustomDatasetWeighted(Dataset):
    def __init__(self, gh, h, classifications, higher_weight=1.0, lower_weight=1.0):
        self.gh = gh
        self.h = h
        self.classifications = classifications

        # Create a list of all pairs
        self.all_pairs = list(itertools.product(range(self.gh.shape[0]), repeat=2))

        # Flatten the classification tensor and calculate weights
        flat_classifications = self.classifications.flatten()
        self.weights = torch.where(
            flat_classifications == 1, higher_weight, lower_weight
        )

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        # Choose a pair based on weighted sampling
        pair_idx = np.random.choice(len(self.all_pairs), p=self.weights.numpy())
        idx1, idx2 = self.all_pairs[pair_idx]
        while idx1 == idx2:
            pair_idx = np.random.choice(len(self.all_pairs), p=self.weights.numpy())
            idx1, idx2 = self.all_pairs[pair_idx]

        gh1, h1 = self.gh[idx1], self.h[idx1]
        gh2, h2 = self.gh[idx2], self.h[idx2]

        # y_inds contains the indices of the chosen samples
        y_inds = torch.tensor([idx1, idx2], dtype=torch.long)

        return gh1, h1, gh2, h2, y_inds


class CustomDataset(Dataset):
    def __init__(self, gh, h, inds):
        self.gh = gh
        self.h = h
        self.n_samples = gh.shape[0]
        self.inds = inds

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Randomly select two different samples
        idx1 = np.random.choice(self.inds)
        idx2 = np.random.choice(self.inds)
        while idx1 == idx2:
            idx2 = np.random.choice(self.inds)

        gh1, h1 = self.gh[idx1], self.h[idx1]
        gh2, h2 = self.gh[idx2], self.h[idx2]

        # y_inds contains the indices of the chosen samples
        y_inds = torch.tensor([idx1, idx2], dtype=torch.long)

        return gh1, h1, gh2, h2, y_inds


def build_weighted_sets(
    num_samples,
    GH_sum,
    H_sum,
    classifications,
    weight_train=10.0,
    weight_test=1.0,
    train_split=0.8,
):
    train_inds = np.random.choice(
        range(num_samples), (int(num_samples * train_split)), replace=False
    )  # train set is 80 percent of data
    test_inds = np.setdiff1d(np.arange(num_samples), train_inds)

    y_lookup_train = classifications[train_inds][:, train_inds]
    y_lookup_test = classifications[test_inds][:, test_inds]

    train_dataset = CustomDatasetWeighted(GH_sum, H_sum, y_lookup_train, weight_train)
    test_dataset = CustomDatasetWeighted(GH_sum, H_sum, y_lookup_test, weight_test)

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    return train_dataloader, test_dataloader, y_lookup_train, y_lookup_test


def build_sets(num_samples, GH_sum, H_sum, train_split=0.8):
    train_inds = np.random.choice(
        range(num_samples), (int(num_samples * train_split)), replace=False
    )  # train set is 80 percent of data
    test_inds = np.setdiff1d(np.arange(num_samples), train_inds)

    train_dataset = CustomDataset(GH_sum, H_sum, train_inds)
    test_dataset = CustomDataset(GH_sum, H_sum, test_inds)

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    return train_dataloader, test_dataloader


class ModelTrainer:
    def __init__(
        self,
        train_dataloader,
        test_dataloader,
        epochs,
        loss_fn,
        optim,
        samps_in_df_train,
        y_lookup_train,
        score_lookup_train,
        samps_in_df_test=None,
        y_lookup_test=None,
        score_lookup_test=None,
        test_best_max=30,
        test_epochs=5,
        epoch_freq=10,
        logits=True,
        subset=False,
        pp=False,
        test_delay=0,
    ):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optim = optim
        self.samps_in_df = samps_in_df_train
        self.y_lookup = y_lookup_train
        self.score_lookup = score_lookup_train if pp else None
        self.samps_in_df_test = (
            samps_in_df_test if samps_in_df_test is not None else samps_in_df_train
        )
        self.y_lookup_test = (
            y_lookup_test if y_lookup_test is not None else y_lookup_train
        )
        self.score_lookup_test = (
            score_lookup_test if score_lookup_test is not None else score_lookup_train
        )
        self.logits = logits
        self.subset = subset

        self.losses = []
        self.test_losses = []
        self.test_loss_acc = 1.0 if subset else 0.5
        self.p_acc = 0.5
        self.r_acc = 0.5
        self.f1_acc = 0.5
        self.best_score = 1.0
        self.test_loss_best = 1.0
        self.test_best_count = 0
        self.test_delay = test_delay
        self.test_best_max = test_best_max
        self.epoch_freq = epoch_freq
        self.test_epochs = test_epochs
        self.acc_losses = [1.0] if subset else [0.5]
        self.p_all = []
        self.r_all = []
        self.f1_all = []

    def train(self, model, plot=True, verbose=True):
        self.best_model_state = model.state_dict()

        for epoch in range(self.epochs):
            model.train()
            for gh1, h1, gh2, h2, y_inds in self.train_dataloader:
                if self.subset:
                    inds_in_df = self.samps_in_df[y_inds[:, 0], y_inds[:, 1]]
                    y_inds_in_df = y_inds[inds_in_df]

                    gh1 = gh1[inds_in_df]
                    h1 = h1[inds_in_df]
                    gh2 = gh2[inds_in_df]
                    h2 = h2[inds_in_df]
                    y = self.y_lookup[y_inds_in_df[:, 0], y_inds_in_df[:, 1]]
                    if self.score_lookup is not None:
                        pp_score = self.score_lookup[
                            y_inds_in_df[:, 0], y_inds_in_df[:, 1]
                        ]
                    else:
                        pp_score = None
                else:
                    y = self.y_lookup[y_inds[:, 0], y_inds[:, 1]]
                    if self.score_lookup is not None:
                        pp_score = self.score_lookup[y_inds[:, 0], y_inds[:, 1]]
                    else:
                        pp_score = None

                self.optim.zero_grad()
                forward_out = model(gh1, h1, gh2, h2, pp_score).squeeze()

                loss = self.loss_fn(forward_out, y)
                loss.backward()
                self.optim.step()

                self.losses.append(loss.detach().numpy())
                self.acc_losses.append(
                    0.95 * self.acc_losses[-1] + 0.05 * loss.detach().numpy()
                )

            if epoch < self.test_delay:
                if epoch % self.epoch_freq == 0:
                    print(f"EPOCH : {epoch} | TRAIN LOSS : {self.acc_losses[-1]}")
            else:
                if epoch % self.epoch_freq == 0:
                    for test_epoch in range(self.test_epochs):
                        with torch.inference_mode():
                            model.eval()
                            batch_test_losses = 0.0
                            p_l = []
                            r_l = []
                            f1_l = []
                            for (
                                gh1_test,
                                h1_test,
                                gh2_test,
                                h2_test,
                                y_inds_test,
                            ) in self.test_dataloader:
                                if self.subset:
                                    inds_in_df_test = self.samps_in_df_test[
                                        y_inds_test[:, 0], y_inds_test[:, 1]
                                    ]
                                    y_inds_in_df_test = y_inds_test[inds_in_df_test]

                                    gh1_test = gh1_test[inds_in_df_test]
                                    h1_test = h1_test[inds_in_df_test]
                                    gh2_test = gh2_test[inds_in_df_test]
                                    h2_test = h2_test[inds_in_df_test]
                                    y_test = self.y_lookup[
                                        y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                                    ]
                                    if self.score_lookup is not None:
                                        pp_score_test = self.score_lookup[
                                            y_inds_in_df_test[:, 0],
                                            y_inds_in_df_test[:, 1],
                                        ]
                                    else:
                                        pp_score_test = None
                                else:
                                    y_test = self.y_lookup_test[
                                        y_inds_test[:, 0], y_inds_test[:, 1]
                                    ]
                                    if self.score_lookup is not None:
                                        pp_score_test = self.score_lookup_test[
                                            y_inds_test[:, 0], y_inds_test[:, 1]
                                        ]
                                    else:
                                        pp_score_test = None

                                forward_out_test = model(
                                    gh1_test,
                                    h1_test,
                                    gh2_test,
                                    h2_test,
                                    pp_score_test,
                                ).squeeze()
                                batch_test_loss = self.loss_fn(forward_out_test, y_test)
                                batch_test_losses += batch_test_loss.item()

                                if self.logits:
                                    forward_out_test = torch.sigmoid(forward_out_test)

                                preds = (forward_out_test >= 0.5).float()
                                p_b, r_b, f1_b = evaluate_matrix(
                                    y_test.numpy(), preds.numpy()
                                )
                                p_l.append(p_b)
                                r_l.append(r_b)
                                f1_l.append(f1_b)

                        test_loss = batch_test_losses / len(self.test_dataloader)
                        p = sum(p_l) / len(self.test_dataloader)
                        r = sum(r_l) / len(self.test_dataloader)
                        f1 = sum(f1_l) / len(self.test_dataloader)

                        self.test_loss_acc = 0.1 * test_loss + 0.9 * self.test_loss_acc
                        self.p_acc = 0.1 * p + 0.9 * self.p_acc
                        self.r_acc = 0.1 * r + 0.9 * self.r_acc
                        self.f1_acc = 0.1 * f1 + 0.9 * self.f1_acc

                        self.test_losses.append(self.test_loss_acc)
                        self.p_all.append(self.p_acc)
                        self.r_all.append(self.r_acc)
                        self.f1_all.append(self.f1_acc)

                        score = (
                            self.p_acc**1.15 * self.r_acc**0.85 * self.f1_acc
                        ) / self.test_loss_acc

                    if self.test_loss_best > self.test_loss_acc:
                        self.test_loss_best = self.test_loss_acc
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model_state = copy.deepcopy(model.state_dict())
                        self.test_best_count = 0
                    else:
                        self.test_best_count += 1
                        if self.test_best_count > self.test_best_max:
                            if plot:
                                plt.figure(figsize=(10, 5))
                                plt.subplot(1, 2, 1)
                                plt.plot(self.p_all, label="Precision")
                                plt.plot(self.r_all, label="Recall")
                                plt.plot(self.f1_all, label="F1")
                                plt.title("Eval")
                                plt.legend()

                                plt.subplot(1, 2, 2)
                                plt.plot(
                                    self.acc_losses, color="red", label="Train Loss"
                                )
                                plt.plot(
                                    self.test_losses, color="green", label="Test Loss"
                                )
                                plt.title("Losses")
                                plt.legend()

                                plt.tight_layout()
                                plt.show()

                            return self.best_model_state
                    print(
                        "P, R, F1 : ",
                        self.p_acc,
                        self.r_acc,
                        self.f1_acc,
                    )
                    print(
                        f"EPOCH : {epoch} | TRAIN LOSS : {self.acc_losses[-1]} | TEST LOSS : {self.test_loss_acc} | BEST : {self.test_loss_best} {self.best_score}"
                    )
                    print()

        if plot:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.plot(self.p_all, label="Precision")
            plt.plot(self.r_all, label="Recall")
            plt.plot(self.f1_all, label="F1")
            plt.title("Eval")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.acc_losses, color="red", label="Train Loss")
            plt.plot(self.test_losses, color="green", label="Test Loss")
            plt.title("Losses")
            plt.legend()

            plt.tight_layout()
            plt.show()

        return self.best_model_state

    def plot_model_curves(self, model, train_or_test="test", num_iter=10):
        p_l = []
        r_l = []
        f1_l = []
        out = []
        y_test_all = []
        dataloader = (
            self.train_dataloader if train_or_test == "train" else self.test_dataloader
        )
        with torch.inference_mode():
            model.eval()
            for e in range(num_iter):
                for (
                    gh1_test,
                    h1_test,
                    gh2_test,
                    h2_test,
                    y_inds_test,
                ) in dataloader:
                    if self.subset:
                        if train_or_test != "train":
                            inds_in_df_test = self.samps_in_df_test[
                                y_inds_test[:, 0], y_inds_test[:, 1]
                            ]
                        else:
                            inds_in_df_test = self.samps_in_df[
                                y_inds_test[:, 0], y_inds_test[:, 1]
                            ]
                        y_inds_in_df_test = y_inds_test[inds_in_df_test]

                        gh1_test = gh1_test[inds_in_df_test]
                        h1_test = h1_test[inds_in_df_test]
                        gh2_test = gh2_test[inds_in_df_test]
                        h2_test = h2_test[inds_in_df_test]
                        if train_or_test != "train":
                            y_test = self.y_lookup_test[
                                y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                            ]
                            if self.score_lookup is not None:
                                pp_score_test = self.score_lookup_test[
                                    y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                                ]
                            else:
                                pp_score_test = None
                        else:
                            y_test = self.y_lookup[
                                y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                            ]
                            if self.score_lookup is not None:
                                pp_score_test = self.score_lookup[
                                    y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                                ]
                            else:
                                pp_score_test = None
                    else:
                        if train_or_test != "train":
                            y_test = self.y_lookup_test[
                                y_inds_test[:, 0], y_inds_test[:, 1]
                            ]
                            if self.score_lookup is not None:
                                pp_score_test = self.score_lookup_test[
                                    y_inds_test[:, 0], y_inds_test[:, 1]
                                ]
                            else:
                                pp_score_test = None
                        else:
                            y_test = self.y_lookup[y_inds_test[:, 0], y_inds_test[:, 1]]
                            if self.score_lookup is not None:
                                pp_score_test = self.score_lookup[
                                    y_inds_test[:, 0], y_inds_test[:, 1]
                                ]
                            else:
                                pp_score_test = None

                    if self.logits:
                        forward_out_test = torch.sigmoid(
                            model(
                                gh1_test,
                                h1_test,
                                gh2_test,
                                h2_test,
                                pp_score_test,
                            ).squeeze()
                        )
                    else:
                        forward_out_test = model(
                            gh1_test,
                            h1_test,
                            gh2_test,
                            h2_test,
                            pp_score_test,
                        ).squeeze()
                    out.append(forward_out_test)
                    y_test_all.append(y_test)

                    preds = (forward_out_test >= 0.5).float()
                    p_b, r_b, f1_b = evaluate_matrix(y_test.numpy(), preds.numpy())
                    p_l.append(p_b)
                    r_l.append(r_b)
                    f1_l.append(f1_b)

        p, r, f1 = sum(p_l) / len(p_l), sum(r_l) / len(r_l), sum(f1_l) / len(f1_l)
        y_t, o_t = [], []
        y_f, o_f = [], []
        out = torch.cat(out)
        y_test_all = torch.cat(y_test_all)
        for i in range(out.shape[0]):
            if y_test_all[i] > 0.0:
                y_t.append(y_test_all[i].item())
                o_t.append(out[i].item())
            else:
                y_f.append(y_test_all[i].item())
                o_f.append(out[i].item())

        sns.distplot(
            o_t,
            hist=False,
            kde=True,
            kde_kws={"shade": True, "linewidth": 3},
            label="True Phenos Match",
        )

        sns.distplot(
            o_f,
            hist=False,
            kde=True,
            kde_kws={"shade": True, "linewidth": 3},
            label="False Phenos Match",
        )

        plt.legend(prop={"size": 10})
        plt.title(f"Model {model.hidden_len}, {model.dropout1.p} Distplot")
        plt.xlabel("Value")
        plt.ylabel("Density")

        plt.show()
        ks_statistic, p_value = ks_2samp(o_t, o_f)

        return p, r, f1, ks_statistic


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)  # + 0.001 * torch.norm(inputs - 0.5)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


def train_full(
    model,
    train_dataloader,
    test_dataloader,
    epochs,
    loss_fn,
    optim,
    samps_in_df,
    y_lookup,
    score_lookup,
    subset=True,
    logits=False,
    test_best_max=100,
):
    losses = []
    test_losses = []
    test_loss_acc = 1.0 if subset else 0.5
    p_acc = 0.5
    r_acc = 0.5
    f1_acc = 0.5
    test_loss_best = 1.0
    test_best_count = 0
    test_best_max = test_best_max
    test_epochs = 20 if subset else 5
    best_model_state = model.state_dict()
    acc_losses = [1.0] if subset else [0.5]
    p_all = []
    r_all = []
    f1_all = []

    for epoch in range(epochs):
        model.train()
        for gh1, h1, gh2, h2, y_inds in train_dataloader:
            if subset:
                inds_in_df = samps_in_df[y_inds[:, 0], y_inds[:, 1]]
                y_inds_in_df = y_inds[inds_in_df]

                gh1 = gh1[inds_in_df]
                h1 = h1[inds_in_df]
                gh2 = gh2[inds_in_df]
                h2 = h2[inds_in_df]
                y = y_lookup[y_inds_in_df[:, 0], y_inds_in_df[:, 1]]
                pp_score = score_lookup[y_inds_in_df[:, 0], y_inds_in_df[:, 1]]
            else:
                y = y_lookup[y_inds[:, 0], y_inds[:, 1]]
                pp_score = score_lookup[y_inds[:, 0], y_inds[:, 1]]

            optim.zero_grad()
            forward_out = model(gh1, h1, gh2, h2, pp_score.unsqueeze(1)).squeeze()

            loss = loss_fn(forward_out, y)
            loss.backward()
            optim.step()

            losses.append(loss.detach().numpy())
            acc_losses.append(0.99 * acc_losses[-1] + 0.01 * loss.detach().numpy())

            if epoch % 10 == 0:
                for test_epoch in range(test_epochs):
                    with torch.inference_mode():
                        model.eval()
                        batch_test_losses = 0.0
                        p_l = []
                        r_l = []
                        f1_l = []
                        for (
                            gh1_test,
                            h1_test,
                            gh2_test,
                            h2_test,
                            y_inds_test,
                        ) in test_dataloader:
                            if subset:
                                inds_in_df_test = samps_in_df[
                                    y_inds_test[:, 0], y_inds_test[:, 1]
                                ]
                                y_inds_in_df_test = y_inds_test[inds_in_df_test]

                                gh1_test = gh1_test[inds_in_df_test]
                                h1_test = h1_test[inds_in_df_test]
                                gh2_test = gh2_test[inds_in_df_test]
                                h2_test = h2_test[inds_in_df_test]
                                y_test = y_lookup[
                                    y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                                ]
                                pp_score_test = score_lookup[
                                    y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                                ]
                            else:
                                y_test = y_lookup[y_inds_test[:, 0], y_inds_test[:, 1]]
                                pp_score_test = score_lookup[
                                    y_inds_test[:, 0], y_inds_test[:, 1]
                                ]

                            forward_out_test = model(
                                gh1_test,
                                h1_test,
                                gh2_test,
                                h2_test,
                                pp_score_test.unsqueeze(1),
                            ).squeeze()
                            batch_test_loss = loss_fn(forward_out_test, y_test)
                            batch_test_losses += batch_test_loss.item()

                            if logits:
                                forward_out_test = torch.sigmoid(forward_out_test)

                            preds = (forward_out_test >= 0.5).float()
                            p_b, r_b, f1_b = evaluate_matrix(
                                y_test.numpy(), preds.numpy()
                            )
                            p_l.append(p_b)
                            r_l.append(r_b)
                            f1_l.append(f1_b)

                        test_loss = batch_test_losses / len(test_dataloader)
                        p = sum(p_l) / len(test_dataloader)
                        r = sum(r_l) / len(test_dataloader)
                        f1 = sum(f1_l) / len(test_dataloader)

                        test_loss_acc = 0.1 * test_loss + 0.9 * test_loss_acc
                        p_acc = 0.1 * p + 0.9 * p_acc
                        r_acc = 0.1 * r + 0.9 * r_acc
                        f1_acc = 0.1 * f1 + 0.9 * f1_acc

                        test_losses.append(test_loss_acc)
                        p_all.append(p_acc)
                        r_all.append(r_acc)
                        f1_all.append(f1_acc)

                if test_loss_acc < test_loss_best:
                    test_loss_best = test_loss_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    test_best_count = 0
                else:
                    test_best_count += 1
                    if test_best_count > test_best_max:
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        plt.plot(p_all, label="Precision")
                        plt.plot(r_all, label="Recall")
                        plt.plot(f1_all, label="F1")
                        plt.plot(test_losses, label="Test Loss")
                        plt.title("Eval")
                        plt.legend()

                        plt.subplot(1, 2, 2)
                        plt.plot(acc_losses, color="red", label="Losses")
                        plt.title("Losses")
                        plt.legend()

                        plt.tight_layout()
                        plt.show()

                        return best_model_state

        if epoch % 10 == 0:
            print(
                "P, R, F1 : ",
                p_acc,
                r_acc,
                f1_acc,
            )

        if epoch % 10 == 0:
            print(
                f"EPOCH : {epoch} | TRAIN LOSS : {acc_losses[-1]} | TEST LOSS : {test_loss_acc} | BEST : {test_loss_best}"
            )
            print()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(p_all, label="Precision")
    plt.plot(r_all, label="Recall")
    plt.plot(f1_all, label="F1")
    plt.title("Eval")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_losses, color="red", label="Train Loss")
    plt.plot(test_losses, color="green", label="Test Loss")
    plt.title("Losses")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return best_model_state
