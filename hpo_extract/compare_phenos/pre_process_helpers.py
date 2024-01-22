import json

import torch
import pandas as pd
import numpy as np

from hpo_extract.setup_data import HPO_DF, HPO_DAG


def hpo_id_to_lbl(hpo_list):
    if type(hpo_list) != list:
        hpo_list = list(hpo_list)
    n_list = []
    for h in hpo_list:
        n_list.append("http://purl.obolibrary.org/obo/" + h.replace(":", "_"))

    lbls = HPO_DF[HPO_DF["id"].isin(n_list)]["lbl"].to_list()
    return lbls


def hpo_lbl_to_id(hpo_list):
    if type(hpo_list) != list:
        hpo_list = list(hpo_list)
    n_list = []
    ids = HPO_DF[HPO_DF["lbl"].isin(hpo_list)]["id"].to_list()
    for i in ids:
        n_list.append(i.strip("http://purl.obolibrary.org/obo/").replace("_", ":"))

    return n_list


def propegate_terms(hpo_list):
    new_terms = set()
    lbl_list = hpo_id_to_lbl(hpo_list)
    for t in lbl_list:
        upper = HPO_DAG.get_ancestors(HPO_DF[HPO_DF["lbl"] == t]["id"].values[0])
        t_set = set(HPO_DF[HPO_DF["id"].isin(upper)]["lbl"].values)
        new_terms.update(t_set)
    new_terms.update(set(lbl_list))

    return hpo_lbl_to_id(list(new_terms - {"Phenotypic abnormality", "All"}))


def add_hpo_ratio(df):
    total_unique_hpo_terms = df["hpo_name"].nunique()
    hpo_counts = df.groupby("gene_symbol")["hpo_name"].nunique()
    hpo_ratio = hpo_counts / total_unique_hpo_terms
    df["hpo_term_ratio"] = df["gene_symbol"].map(hpo_ratio)
    return df


def filter_small(all_sample_phenos):
    new_all_sample_phenos = {}
    for samp_num, samp in all_sample_phenos.items():
        if len(samp["hpo_terms"]) > 1:
            new_all_sample_phenos[samp_num] = samp

    return new_all_sample_phenos


def map_frequency(value):
    hp_mapping = {
        "HP:0040283": 0.5,
        "HP:0040281": 0.9,
        "HP:0040282": 0.7,
        "HP:0040284": 0.1,
        "HP:0040280": 1.0,
        "HP:0040285": 0.0,
    }
    if value == "-":
        return 0.5
    elif str(value).startswith("HP:"):
        return hp_mapping.get(value, "Unknown")
    else:
        try:
            x, y = value.split("/")
            return float(x) / float(y)
        except:
            return "Unknown"


def load_gene_data(file_path):
    with open(file_path, "r") as file:
        data = file.read()

    json_data = json.loads(data)

    columns = []
    rows = []

    for entry in json_data:
        for key, value in entry.items():
            if not columns:
                columns = key.split("\t")
            row_data = value.split("\t")
            rows.append(row_data)

    df = pd.DataFrame(rows, columns=columns)
    return df


def load_genes_for_samples(path):
    full_gene_df = load_gene_data(path).drop(
        ["ncbi_gene_id", "hpo_id", "disease_id"], axis=1
    )

    gene_df = add_hpo_ratio(full_gene_df)
    gene_df["frequency"] = gene_df["frequency"].apply(map_frequency)
    gene_df["frequency"] = pd.to_numeric(gene_df["frequency"], errors="coerce")
    gene_df = gene_df.dropna(subset=["frequency"])
    gene_df.reset_index(drop=True, inplace=True)

    mean_values = (
        gene_df.groupby(["gene_symbol", "hpo_name"])["frequency"].mean().reset_index()
    )
    df_merged = pd.merge(
        gene_df, mean_values, on=["gene_symbol", "hpo_name"], suffixes=("", "_mean")
    )
    df_merged["frequency"] = df_merged["frequency_mean"]
    df_merged.drop(columns=["frequency_mean"], inplace=True)
    all_gene_ass_deduped = df_merged.drop_duplicates(subset=["gene_symbol", "hpo_name"])
    all_gene_ass_deduped = all_gene_ass_deduped[
        all_gene_ass_deduped["gene_symbol"] != "-"
    ].reset_index(drop=True)
    return all_gene_ass_deduped


def make_hpo_list(sample_phenos_df):
    hpo_terms_list = []
    for entry in sample_phenos_df["hpo_ids"]:
        for t in entry:
            if t not in hpo_terms_list:
                hpo_terms_list.append(t)
    return list(set(hpo_terms_list))


def make_hpo_t(all_sample_phenos, hpo_list):
    samples = []
    gene_name_list = []
    for key in all_sample_phenos.keys():
        samples.append(all_sample_phenos[key]["hpo_terms"])
        gene_name_list.append(all_sample_phenos[key]["gene"])

    tensor = torch.zeros((len(samples), len(hpo_list)))

    for i, sample in enumerate(samples):
        for term in sample:
            if term in hpo_list:
                j = hpo_list.index(term)
                tensor[i, j] = 1.0

    return tensor, gene_name_list


def make_hpo_t_with_ids(sample_phenos_df, hpo_list):
    samples = sample_phenos_df["hpo_ids"].tolist()

    tensor = torch.zeros((len(samples), len(hpo_list)))

    for i, sample in enumerate(samples):
        for term in sample:
            if term in hpo_list:
                j = hpo_list.index(term)
                tensor[i, j] = 1.0

    return tensor


def make_hpo_ic(hpo_t):
    return -torch.log2(torch.mean(hpo_t, dim=0))


def make_gene_freq(all_gene_ass, hpo_l):
    hpo_list_lbls = []
    for h in hpo_l:
        hpo_list_lbls.append(hpo_id_to_lbl([h])[0])

    unique_genes = all_gene_ass["gene_symbol"].unique()

    grouped_df = (
        all_gene_ass.groupby(["gene_symbol", "hpo_name"])["frequency"]
        .sum()
        .reset_index()
    )
    pivot_df = grouped_df.pivot(
        index="gene_symbol", columns="hpo_name", values="frequency"
    )
    pivot_df = pivot_df.reindex(index=unique_genes, columns=hpo_list_lbls, fill_value=0)
    frequency_tensor = torch.tensor(np.nan_to_num(pivot_df.to_numpy()))

    hpo_to_genes_tensor = (frequency_tensor != 0).T

    return frequency_tensor, hpo_to_genes_tensor, unique_genes


def make_gene_ic(all_gene_ass, unique_genes, hpo_l):
    hpo_list_lbls = []
    for h in hpo_l:
        hpo_list_lbls.append(hpo_id_to_lbl([h])[0])

    ics_per_gene = []
    all_genes_in_data = all_gene_ass[all_gene_ass["hpo_name"].isin(hpo_list_lbls)][
        "gene_symbol"
    ].to_list()
    for ug in unique_genes:
        ics_per_gene.append(all_genes_in_data.count(ug) + 1)
    ics_per_gene = torch.tensor(ics_per_gene).float()
    ics_per_gene = ics_per_gene / len(hpo_list_lbls)
    return -torch.log2(ics_per_gene)


def make_truth_inds(ind_key, genes_list):
    truth_inds = []
    for inds in ind_key:
        if genes_list[inds[0]] == genes_list[inds[1]]:
            truth_inds.append(1)
        else:
            truth_inds.append(0)

    truth_inds = torch.tensor(truth_inds).float()
    return truth_inds


def make_gene_inputs(all_gene_ass, hpo_t, hpo_l):
    frequency_tensor, hpo_to_genes_tensor, unique_genes = make_gene_freq(
        all_gene_ass, hpo_l
    )
    input_gene_asses = hpo_to_genes_tensor.float().T @ hpo_t.T
    gene_ic = make_gene_ic(all_gene_ass, unique_genes, hpo_l)
    gene_inputs = input_gene_asses.T * gene_ic
    return gene_inputs / gene_inputs.mean(dim=1).unsqueeze(1)


def make_hpo_inputs(all_gene_ass, hpo_t, hpo_l):
    frequency_tensor, hpo_to_genes_tensor, unique_genes = make_gene_freq(
        all_gene_ass, hpo_l
    )
    hpo_ic = make_hpo_ic(hpo_t)
    hpo_inputs = hpo_t * hpo_ic
    hpo_inputs = hpo_inputs / hpo_inputs.mean(dim=1).unsqueeze(1)
    gene_ic = make_gene_ic(all_gene_ass, unique_genes, hpo_l)
    gene_freq_ic = frequency_tensor * gene_ic.unsqueeze(0).T
    return hpo_inputs, gene_freq_ic


def make_gene_inputs_gen(gene_inputs, num_samples, chunk_size=16):
    for gene_inputs_chunk_inds in range(0, num_samples, chunk_size):
        yield gene_inputs[gene_inputs_chunk_inds : gene_inputs_chunk_inds + chunk_size]


def make_hpo_inputs_gen(hpo_inputs, gene_freq_ic, num_samples, chunk_size=16):
    for hpo_inputs_chunk_inds in range(0, hpo_inputs.shape[0], chunk_size):
        hpo_inputs_chunk = hpo_inputs[
            hpo_inputs_chunk_inds : hpo_inputs_chunk_inds + chunk_size
        ].unsqueeze(1) * gene_freq_ic.unsqueeze(0)
        yield hpo_inputs_chunk


def make_gene_inputs_gen_from_df(gene_inputs, sample_ids, chunk_size):
    for gene_inputs_chunk_inds in range(0, len(sample_ids), chunk_size):
        yield gene_inputs[
            gene_inputs_chunk_inds : gene_inputs_chunk_inds + chunk_size
        ], sample_ids[gene_inputs_chunk_inds : gene_inputs_chunk_inds + chunk_size]


def make_hpo_inputs_gen_from_df(hpo_inputs, gene_freq_ic, sample_ids, chunk_size):
    for hpo_inputs_chunk_inds in range(0, hpo_inputs.shape[0], chunk_size):
        hpo_inputs_chunk = hpo_inputs[
            hpo_inputs_chunk_inds : hpo_inputs_chunk_inds + chunk_size
        ].unsqueeze(1) * gene_freq_ic.unsqueeze(0)
        yield hpo_inputs_chunk, sample_ids[
            hpo_inputs_chunk_inds : hpo_inputs_chunk_inds + chunk_size
        ]


def df_with_ids(df, sampl_id_list):
    def find_index(entity_id, sampl_id_list):
        try:
            return sampl_id_list.index(entity_id)
        except ValueError:
            return None

    df["query_index"] = (
        df["#query"].astype("str").apply(lambda x: find_index(x, sampl_id_list))
    )
    df["entity_index"] = (
        df["entity_id"].astype("str").apply(lambda x: find_index(x, sampl_id_list))
    )

    return df


def filter_dataframe(df, n):
    excluded_ids = np.random.choice(df["entity_id"].unique(), n, replace=False)
    df_filtered = df[
        ~df["#query"].isin(excluded_ids) & ~df["entity_id"].isin(excluded_ids)
    ]
    df_excluded = df[
        df["#query"].isin(excluded_ids) & df["entity_id"].isin(excluded_ids)
    ]

    return df_filtered, df_excluded, excluded_ids.tolist()


def make_X_train_from_df(
    hpo_inputs, gene_freq_ic, gene_inputs, genes_list, df, sample_ids, chunk_size=16
):
    gene_inputs_gen = make_gene_inputs_gen_from_df(gene_inputs, sample_ids, chunk_size)
    hpo_inputs_gen = make_hpo_inputs_gen_from_df(
        hpo_inputs, gene_freq_ic, sample_ids, chunk_size
    )
    key = []

    for i, (
        (gene_inputs_chunk, sample_ids_chunk),
        (hpo_inputs_chunk, h_sample_ids_chunk),
    ) in enumerate(zip(gene_inputs_gen, hpo_inputs_gen)):
        print(f"{i} / {int(len(sample_ids)/chunk_size)}")
        gene_inputs_gen2 = make_gene_inputs_gen_from_df(
            gene_inputs, sample_ids, chunk_size
        )
        hpo_inputs_gen2 = make_hpo_inputs_gen_from_df(
            hpo_inputs, gene_freq_ic, sample_ids, chunk_size
        )
        combined_tensor = torch.cat(
            (gene_inputs_chunk.unsqueeze(-1), hpo_inputs_chunk), dim=-1
        )  # (b, g_l, h_l+1)
        queries_df = df[df["#query"].astype("str").isin(sample_ids_chunk)]

        for i2, (
            (gene_inputs_chunk2, sample_ids_chunk2),
            (hpo_inputs_chunk2, h_sample_ids_chunk2),
        ) in enumerate(zip(gene_inputs_gen2, hpo_inputs_gen2)):
            combined_tensor2 = torch.cat(
                (gene_inputs_chunk2.unsqueeze(-1), hpo_inputs_chunk2), dim=-1
            )  # (b, g_l, h_l+1)

            sliced_df = queries_df[
                queries_df["entity_id"].astype("str").isin(sample_ids_chunk2)
            ]
            real_inds = sliced_df["query_index"].to_numpy()
            real_inds2 = sliced_df["entity_index"].to_numpy()
            inds = real_inds - chunk_size * i
            inds2 = real_inds2 - chunk_size * i2

            X_train_chunk = torch.stack(
                [combined_tensor[inds], combined_tensor2[inds2]], dim=1
            )  # (b*b/2ish, 2, g_l, h_l+1)
            real_inds_key = [(a, b) for a, b in zip(real_inds, real_inds2)]
            y_chunk = make_truth_inds(real_inds_key, genes_list)
            assert (y_chunk == torch.tensor(sliced_df["same"].tolist()).float()).all()
            assert (
                y_chunk
                == torch.tensor(
                    df.loc[sliced_df.index.tolist()]["same"].tolist()
                ).float()
            ).all()

            key.extend(sliced_df.index.tolist())
            h_sum_batch = torch.sum(X_train_chunk, dim=-2)  # (b*b, 2, h_l+1)

            if i == 0 and i2 == 0:
                gh_sum = torch.sum(
                    X_train_chunk[:, :, :, 0:1] * X_train_chunk[:, :, :, 1:], dim=-2
                )
                h_sum = h_sum_batch[:, :, 1:]
                g_sum = h_sum_batch[:, :, 0:1]
                y_all = y_chunk
            else:
                g_sum = torch.cat((g_sum, h_sum_batch[:, :, 0:1]), dim=0)  # (b, 2, 1)
                h_sum = torch.cat((h_sum, h_sum_batch[:, :, 1:]), dim=0)  # (b, 2, h_l)
                gh_sum = torch.cat(
                    (
                        gh_sum,
                        torch.sum(
                            X_train_chunk[:, :, :, 0:1] * X_train_chunk[:, :, :, 1:],
                            dim=-2,
                        ),
                    ),
                    dim=0,
                )
                y_all = torch.cat((y_all, y_chunk))

    return gh_sum, h_sum, g_sum, y_all, key


def load_sample_phenos(all_sample_phenos_path, filter_sm=True):
    with open(all_sample_phenos_path, "r") as f:
        all_sample_phenos = json.load(f)
    if filter_sm:
        all_sample_phenos = filter_small(all_sample_phenos)

    data_for_df = []
    for sample_num, details in all_sample_phenos.items():
        data_for_df.append(
            {
                "sample_num": sample_num,
                "gene": details["gene"],
                "hpo_ids": details["hpo_terms"],
                "hpo_lbls": hpo_id_to_lbl(details["hpo_terms"]),
            }
        )
    return pd.DataFrame(data_for_df)


def load_all_df(
    all_sample_phenos_path, genes_to_phenos_path, comparisons_path, filter_sm=True
):
    sample_phenos_df = load_sample_phenos(all_sample_phenos_path, filter_sm=filter_sm)

    genes_to_phenos_df = load_genes_for_samples(genes_to_phenos_path)

    comparisons_df = pd.read_csv(comparisons_path)

    return sample_phenos_df, genes_to_phenos_df, comparisons_df


def prepare_all_df(sample_phenos_df, comparisons_df, curate=True):
    sample_phenos_df["hpo_ids"] = sample_phenos_df["hpo_ids"].apply(propegate_terms)
    sample_phenos_df["hpo_lbls"] = sample_phenos_df["hpo_lbls"].apply(tuple)
    sample_phenos_unique_df = sample_phenos_df.drop_duplicates(subset=["hpo_ids"])
    sample_phenos_unique_df["hpo_lbls"] = sample_phenos_df["hpo_lbls"].apply(list)
    sample_phenos_df = sample_phenos_unique_df.reset_index(drop=True)
    sample_ids = sample_phenos_df["sample_num"].tolist()

    if curate:
        comparisons_same = comparisons_df[comparisons_df["same"] == True]
        comparisons_different = comparisons_df[comparisons_df["same"] == False]
        comparisons_highest_different = comparisons_different[
            comparisons_different["score"] > 0.1
        ]

        curated_df = pd.concat(
            [comparisons_highest_different, comparisons_same], ignore_index=True
        )
        curated_df = df_with_ids(curated_df, sample_ids)
        curated_df = curated_df.dropna()
        curated_df["query_index"] = curated_df["query_index"].apply(int)
        curated_df["entity_index"] = curated_df["entity_index"].apply(int)
        curated_df = curated_df.drop("Unnamed: 0", axis=1)
        comparisons_df = curated_df.reset_index(drop=True)
    else:
        comparisons_df = df_with_ids(comparisons_df, sample_ids)
        comparisons_df = comparisons_df.dropna()
        comparisons_df["query_index"] = comparisons_df["query_index"].apply(int)
        comparisons_df["entity_index"] = comparisons_df["entity_index"].apply(int)

    return sample_phenos_df, comparisons_df


def make_intermediate_data(sample_phenos_df, genes_to_phenos_df, other_hpo_terms):
    hpo_list = make_hpo_list(sample_phenos_df)
    hpo_set = set(hpo_list)
    hpo_set.update(set(other_hpo_terms))
    hpo_list = list(hpo_set)
    hpo_list_lbls = []
    for h in hpo_list:
        hpo_list_lbls.append(hpo_id_to_lbl([h])[0])

    hpo_tensor = make_hpo_t_with_ids(sample_phenos_df, hpo_list)
    hpo_ic = make_hpo_ic(
        torch.cat([torch.ones(1, hpo_tensor.shape[1]), hpo_tensor])
    )  # cat ones for no nans. at least one hpo term
    hpo_tensor_ic = hpo_tensor * hpo_ic
    hpo_inputs = hpo_tensor_ic / hpo_tensor_ic.mean(dim=1).unsqueeze(
        1
    )  # ([1320, 3567])   (s, h)

    frequency_tensor, hpo_to_genes_tensor, unique_genes = make_gene_freq(
        genes_to_phenos_df, hpo_list
    )
    gene_ic = make_gene_ic(genes_to_phenos_df, unique_genes, hpo_list)
    gene_freq_ic = (
        frequency_tensor * gene_ic.unsqueeze(0).T
    ).float()  # ([4975, 3567])   (g, h)

    gene_inputs = make_gene_inputs(
        genes_to_phenos_df, hpo_tensor, hpo_list
    )  # ([1320, 5005])   (s, g)

    return hpo_tensor, hpo_inputs, gene_inputs, gene_freq_ic, hpo_list, hpo_list_lbls


def make_input_tensors(hpo_inputs, gene_inputs, gene_freq_ic):
    GH_sum = (
        hpo_inputs * torch.matmul(gene_inputs, gene_freq_ic) + hpo_inputs
    )  # ([1320, 3567])   (s, h)
    H_sum = (
        hpo_inputs * torch.matmul(torch.ones_like(gene_inputs), gene_freq_ic)
        + hpo_inputs
    )  # ([1320, 3567]) (s, h)
    return GH_sum, H_sum


def make_y_lookup(num_samples, df):
    y_lookup = torch.zeros((num_samples, num_samples), dtype=torch.float32)
    for i in range(num_samples):
        l = df["entity_index"][df["query_index"] == i].to_numpy()
        y_vals = df["same"][df["query_index"] == i].to_numpy()
        y_lookup[i, l] = torch.tensor(y_vals).float()
        l = df["query_index"][df["entity_index"] == i].to_numpy()
        y_vals = df["same"][df["entity_index"] == i].to_numpy()
        y_lookup[i, l] = torch.tensor(y_vals).float()

    return y_lookup


def make_y_lookup_scratch(sp_df):
    y_lookup_fresh = torch.zeros((len(sp_df), len(sp_df)), dtype=torch.float32)

    for gene in sp_df["gene"].unique().tolist():
        gene_inds = sp_df[sp_df["gene"] == gene].index.tolist()
        rows, cols = torch.meshgrid(
            torch.tensor(gene_inds).int(), torch.tensor(gene_inds).int()
        )
        y_lookup_fresh[rows, cols] = 1.0
    y_lookup_fresh = y_lookup_fresh.fill_diagonal_(0.0)
    return y_lookup_fresh


def make_samps_in_df(num_samples, df):
    samps_in_df1 = torch.zeros((num_samples, num_samples), dtype=torch.bool)
    samps_in_df2 = torch.zeros((num_samples, num_samples), dtype=torch.bool)

    for i in range(num_samples):
        l = df["entity_index"][df["query_index"] == i].to_numpy()
        samps_in_df1[i, l] = True
        l = df["query_index"][df["entity_index"] == i].to_numpy()
        samps_in_df1[i, l] = True
    for i in range(num_samples):
        l = df["entity_index"][df["query_index"] == i].to_numpy()
        samps_in_df2[l, i] = True
        l = df["query_index"][df["entity_index"] == i].to_numpy()
        samps_in_df2[l, i] = True

    return samps_in_df2 & samps_in_df1  # (s, s) true if comparison in curated_df


def make_score_lookup(num_samples, df):
    score_lookup = torch.zeros((num_samples, num_samples), dtype=torch.float32)
    for i in range(num_samples):
        l = df["entity_index"][df["query_index"] == i].to_numpy()
        s_vals = df["score"][df["query_index"] == i].to_numpy()
        score_lookup[i, l] = torch.tensor(s_vals).float()
        l = df["query_index"][df["entity_index"] == i].to_numpy()
        s_vals = df["score"][df["entity_index"] == i].to_numpy()
        score_lookup[i, l] = torch.tensor(s_vals).float()

    return score_lookup
