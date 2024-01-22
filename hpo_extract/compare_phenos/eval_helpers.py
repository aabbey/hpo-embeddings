import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_matrix(true_matrix, pred_matrix):
    """returns the precision, recall, and f1 of two matricies or vectors. numpy or tensor"""
    assert (
        true_matrix.shape == pred_matrix.shape
    ), "Both matrices should have the same shape."
    TP = np.sum((true_matrix == 1.0) & (pred_matrix == 1.0))
    FP = np.sum((true_matrix == 0.0) & (pred_matrix == 1.0))
    TN = np.sum((true_matrix == 0.0) & (pred_matrix == 0.0))
    FN = np.sum((true_matrix == 1.0) & (pred_matrix == 0.0))
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return precision, recall, f1


def plot_model_curves(
    model,
    test_dataloader,
    comparisons_df,
    num_iter,
    samps_in_df,
    y_lookup,
    score_lookup,
    logits=False,
    subset=True,
):
    p_l = []
    r_l = []
    f1_l = []
    out = []
    y_test_all = []
    y_test_inds_all = []
    with torch.inference_mode():
        model.eval()
        for e in range(num_iter):
            for gh1_test, h1_test, gh2_test, h2_test, y_inds_test in test_dataloader:
                if subset:
                    inds_in_df_test = samps_in_df[y_inds_test[:, 0], y_inds_test[:, 1]]
                    y_inds_in_df_test = y_inds_test[inds_in_df_test]

                    gh1_test = gh1_test[inds_in_df_test]
                    h1_test = h1_test[inds_in_df_test]
                    gh2_test = gh2_test[inds_in_df_test]
                    h2_test = h2_test[inds_in_df_test]
                    y_test = y_lookup[y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]]
                    pp_score_test = score_lookup[
                        y_inds_in_df_test[:, 0], y_inds_in_df_test[:, 1]
                    ]
                else:
                    y_test = y_lookup[y_inds_test[:, 0], y_inds_test[:, 1]]
                    pp_score_test = score_lookup[y_inds_test[:, 0], y_inds_test[:, 1]]

                if logits:
                    forward_out_test = torch.sigmoid(
                        model(
                            gh1_test,
                            h1_test,
                            gh2_test,
                            h2_test,
                            pp_score_test.unsqueeze(1),
                        ).squeeze()
                    )
                else:
                    forward_out_test = model(
                        gh1_test, h1_test, gh2_test, h2_test, pp_score_test.unsqueeze(1)
                    ).squeeze()
                out.append(forward_out_test)
                y_test_all.append(y_test)
                if subset:
                    for r in range(len(y_inds_in_df_test.numpy())):
                        i = y_inds_in_df_test.numpy()[r, :]
                        df_inds = comparisons_df[
                            (
                                (comparisons_df["query_index"] == i[0])
                                | (comparisons_df["entity_index"] == i[0])
                            )
                            & (
                                (comparisons_df["entity_index"] == i[1])
                                | (comparisons_df["query_index"] == i[1])
                            )
                        ]["same"]
                        y_test_inds_all.extend(df_inds.index.tolist())
                else:
                    for r in range(len(y_inds_test.numpy())):
                        i = y_inds_test.numpy()[r, :]
                        df_inds = comparisons_df[
                            (
                                (comparisons_df["query_index"] == i[0])
                                | (comparisons_df["entity_index"] == i[0])
                            )
                            & (
                                (comparisons_df["entity_index"] == i[1])
                                | (comparisons_df["query_index"] == i[1])
                            )
                        ]["same"]
                        y_test_inds_all.extend(df_inds.index.tolist())

                preds = (forward_out_test >= 0.5).float()
                p_b, r_b, f1_b = evaluate_matrix(y_test.numpy(), preds.numpy())
                p_l.append(p_b)
                r_l.append(r_b)
                f1_l.append(f1_b)
            p = sum(p_l) / len(test_dataloader)
            r = sum(r_l) / len(test_dataloader)
            f1 = sum(f1_l) / len(test_dataloader)

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
    plt.title(f"Model {model.hidden_len}, {model.dropout.p} Distplot")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.show()
