"""Masked prediction loss for labels that are not observed on every document.

A NaN in a label column of the Corpus marks a document without that label:
`Corpus` puts 0 in `M_labels` and 0 in `M_labels_mask`, and `step_batch` drops
the document from that label's loss (sum of per-document NLL over the observed
documents / batch size, the normalisation of the reconstruction term).

Checks, on a synthetic one-factor corpus with a regression and a binary label:
    (i)   without NaN, the loss components and the gradients equal the values
          of the package before the change (branch learned-embedding-noise-var
          @ d214219, computed by the same construction, hard-coded below) for
          mean_field and for mixture_of_gaussians C = 2;
    (ii)  with NaN, the loss is finite, does not depend on the placeholder of
          the unobserved documents, and splits: loss(mask S) + loss(mask S^c)
          = loss(all observed) label by label, so the unobserved documents have
          no gradient; a label with no observed document in the batch has a
          zero gradient on its head;
    (iii) `labels_in_encoder=True` with a missing label raises;
    (iv)  the Corpus keeps `M_labels` byte-identical without NaN and fills the
          placeholder with NaN present (class 0 for multiclass, `num_classes`
          from the observed values).

Plain script (no pytest in the project env):
    PYTHONPATH=. python tests/test_label_mask.py
(also collected by pytest). ~10 s CPU.
"""

import math
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from deeplatent.corpus import Corpus
from deeplatent.models import IdealPointNN

LINEAR = {"hidden_dims": [], "activation": None, "bias": True, "dropout": 0.0}
LABELS = {"y": {"column": "y", "type": "regression"},
          "b": {"column": "b", "type": "binary"}}
BATCH = range(32)
SEED_DRAW = 123

# Values of the package before the mask (d214219), same corpus, same batch, same
# seeds: loss components, per-label loss, sum |grad| of the head parameters and of
# the encoder. Exact equality up to float32 rounding is the claim of test (i).
REFERENCE = {
    "mean_field": {
        "loss": 12.607163429260254,
        "recon": 10.031082153320312,
        "div": 0.1326427459716797,
        "pred": 2.443438768386841,
        "per_label": {"y": 1.5378857851028442, "b": 0.9055529832839966},
        "grad_predictor": {
            "predictors.y.neural_net.pred_0.weight": 1.0187172889709473,
            "predictors.b.neural_net.pred_0.weight": 0.29162365198135376,
            "predictors.b.neural_net.pred_0.bias": 0.16197168827056885,
            "noise_log_var.y": 0.11894726753234863,
        },
        "grad_encoder_abs_sum": 18.14460039138794,
    },
    "mog_C2": {
        "loss": 11.900308609008789,
        "recon": 9.449752807617188,
        "div": 0.09620635956525803,
        "pred": 2.3543496131896973,
        "per_label": {"y": 1.6382300853729248, "b": 0.716119647026062},
        "grad_predictor": {
            "predictors.y.neural_net.pred_0.weight": 1.104394555091858,
            "predictors.b.neural_net.pred_0.weight": 0.06837093085050583,
            "predictors.b.neural_net.pred_0.bias": 0.10279357433319092,
            "noise_log_var.y": 0.21929150819778442,
        },
        "grad_encoder_abs_sum": 26.977001190185547,
    },
}
FAMILIES = {"mean_field": ("mean_field", 1), "mog_C2": ("mixture_of_gaussians", 2)}


def synthetic(n=64, d=8, seed=0):
    """x = z w + noise, y = 0.8 z + noise, b = 1[z + noise > 0]: the reference."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    w = rng.standard_normal(d) / np.sqrt(d)
    x = np.outer(z, w) + 0.5 * rng.standard_normal((n, d))
    y = 0.8 * z + 0.3 * rng.standard_normal(n)
    b = (z + 0.5 * rng.standard_normal(n) > 0).astype(float)
    df = pd.DataFrame({"doc": [""] * n, "y": y.astype(np.float32),
                       "b": b.astype(np.float32)})
    return df, x.astype(np.float32)


def build(df, x, vi_type, C, labels_in_encoder=False, seed=0):
    corpus = Corpus(
        df,
        modalities={"doc": {"views": {"embedding": {"type": "embedding",
                                                    "matrix": x}}}},
        labels=LABELS,
    )
    model = IdealPointNN(
        corpus, n_ideal_points=1, ae_type="vae", vi_type=vi_type, mixture_components=C,
        update_prior=False, labels_in_encoder=labels_in_encoder, w_pred_loss=1.0,
        encoder_args={"doc_embedding": dict(LINEAR)},
        decoder_args={"doc_embedding": dict(LINEAR)},
        predictor_args={"y": {**LINEAR, "bias": False, "loss_weight": 1.0},
                        "b": {**LINEAR, "loss_weight": 1.0}},
        num_steps=0, batch_size=32, num_workers=0,
        optim_args={"main": {"lr": 1e-3}, "predictor": {"lr": 1e-3}},
        print_every_n_steps=10**9, ckpt_folder=tempfile.mkdtemp(),
        return_best_model=False,
        device="cpu", seed=seed,
    )
    return corpus, model


def one_batch(corpus, model, rows=BATCH):
    """One training step on a fixed batch with a fixed reparameterisation draw."""
    batch = default_collate([corpus[i] for i in rows])
    torch.manual_seed(SEED_DRAW)
    loss = model.step_batch(batch, corpus, validation=False)
    grads = {name: float(p.grad.abs().sum())
             for name, p in model.predictor.named_parameters() if p.grad is not None}
    enc = sum(float(p.grad.abs().sum())
              for p in model.encoder.parameters() if p.grad is not None)
    return {"loss": loss, "recon": float(model.reconstruction_loss.detach()),
            "div": float(model.divergence_loss.detach()),
            "pred": float(model.prediction_loss.detach()),
            "per_label": dict(model.per_label_losses),
            "n_obs": dict(model.per_label_n_obs), "grad_predictor": grads,
            "grad_encoder_abs_sum": enc}


def close(a, b, rel=1e-5, abs_=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def with_nan(df, rows_y, rows_b):
    out = df.copy()
    out.loc[list(rows_y), "y"] = np.nan
    out.loc[list(rows_b), "b"] = np.nan
    return out


def test_no_nan_reproduces_the_package_before_the_mask():
    """(i) all labels observed: loss, per-label losses and gradients unchanged."""
    torch.use_deterministic_algorithms(True)
    df, x = synthetic()
    for family, (vi, C) in FAMILIES.items():
        ref = REFERENCE[family]
        got = one_batch(*build(df, x, vi, C))
        for key in ("loss", "recon", "div", "pred", "grad_encoder_abs_sum"):
            assert close(got[key], ref[key]), (family, key, got[key], ref[key])
        for label, v in ref["per_label"].items():
            assert close(got["per_label"][label], v), (family, label, v)
        for name, v in ref["grad_predictor"].items():
            assert close(got["grad_predictor"][name], v), (family, name, v)
        assert got["n_obs"] == {"y": 32, "b": 32}


def test_masked_loss_splits_over_observed_documents():
    """(ii) loss(S) + loss(S^c) = loss(all) per label; finite; placeholder-free."""
    torch.use_deterministic_algorithms(True)
    df, x = synthetic()
    S_y, S_b = set(range(0, 32, 2)), set(range(0, 12))
    Sc_y, Sc_b = set(range(32)) - S_y, set(range(32)) - S_b
    for family, (vi, C) in FAMILIES.items():
        full = one_batch(*build(df, x, vi, C))
        part = one_batch(*build(with_nan(df, S_y, S_b), x, vi, C))
        rest = one_batch(*build(with_nan(df, Sc_y, Sc_b), x, vi, C))
        assert math.isfinite(part["loss"]) and math.isfinite(rest["loss"])
        assert part["n_obs"] == {"y": 16, "b": 20}
        assert rest["n_obs"] == {"y": 16, "b": 12}
        for label in ("y", "b"):
            s = part["per_label"][label] + rest["per_label"][label]
            assert close(s, full["per_label"][label]), (family, label, s)
        # The value stored for an unobserved document is irrelevant: Corpus puts 0,
        # any other placeholder in M_labels gives the same loss and gradients.
        corpus, model = build(with_nan(df, S_y, S_b), x, vi, C)
        corpus.M_labels[list(S_y), 0] = 1e3
        corpus.M_labels[list(S_b), 1] = 1.0
        alt = one_batch(corpus, model)
        assert close(alt["loss"], part["loss"]) and close(alt["pred"], part["pred"])
        for name, v in part["grad_predictor"].items():
            assert close(alt["grad_predictor"][name], v), (family, name)
        # No observed document for a label in the batch: its head gets no gradient,
        # the other label and the reconstruction are untouched.
        none_b = one_batch(*build(with_nan(df, set(), set(range(32))), x, vi, C))
        assert none_b["n_obs"]["b"] == 0 and none_b["per_label"]["b"] == 0.0
        for name in ("predictors.b.neural_net.pred_0.weight",
                     "predictors.b.neural_net.pred_0.bias"):
            assert none_b["grad_predictor"][name] == 0.0
        assert close(none_b["per_label"]["y"], full["per_label"]["y"])
        assert close(none_b["recon"], full["recon"])


def test_labels_in_encoder_with_missing_labels_raises():
    """(iii) the encoder must not read the placeholder of an unobserved label."""
    df, x = synthetic()
    build(df, x, "mean_field", 1, labels_in_encoder=True)  # all observed: allowed
    try:
        build(with_nan(df, {3}, set()), x, "mean_field", 1, labels_in_encoder=True)
    except ValueError as e:
        assert "labels_in_encoder" in str(e)
    else:
        raise AssertionError("labels_in_encoder=True with a NaN label did not raise")


def test_corpus_labels_and_mask():
    """(iv) M_labels unchanged without NaN; placeholder 0 and mask 0 with NaN."""
    df, x = synthetic()
    views = {"doc": {"views": {"embedding": {"type": "embedding", "matrix": x}}}}
    corpus = Corpus(df, modalities=views, labels=LABELS)
    expected = np.stack([df["y"].to_numpy(np.float32), df["b"].to_numpy(np.float32)],
                        axis=1)
    assert corpus.M_labels.dtype == np.float32
    assert corpus.M_labels.tobytes() == expected.tobytes()
    assert corpus.M_labels_mask.shape == expected.shape
    assert np.all(corpus.M_labels_mask == 1.0)
    assert corpus[0]["M_labels_mask"].shape == (2,)

    df2 = with_nan(df, {1, 5}, {2})
    df2["k"] = (np.arange(len(df)) % 3).astype(float)
    df2.loc[[0, 7], "k"] = np.nan  # class 0 only at rows 0 and 7 -> classes still 0..2
    corpus2 = Corpus(df2, modalities=views,
                     labels={**LABELS, "k": {"column": "k", "type": "multiclass"}})
    assert corpus2.labels_info["k"]["num_classes"] == 3
    assert np.array_equal(corpus2.M_labels_mask[[1, 5], 0], [0.0, 0.0])
    assert corpus2.M_labels_mask[2, 1] == 0.0 and corpus2.M_labels_mask[3, 1] == 1.0
    assert np.array_equal(corpus2.M_labels_mask[[0, 7], 2], [0.0, 0.0])
    assert not np.isnan(corpus2.M_labels).any()
    assert corpus2.M_labels[1, 0] == 0.0 and corpus2.M_labels[0, 2] == 0.0
    assert corpus2.M_labels[3, 0] == df["y"][3] and corpus2.M_labels[3, 2] == 0.0
    assert corpus2.M_labels_mask.sum() == 3 * 64 - 5


if __name__ == "__main__":
    for fn in (test_corpus_labels_and_mask,
               test_labels_in_encoder_with_missing_labels_raises,
               test_no_nan_reproduces_the_package_before_the_mask,
               test_masked_loss_splits_over_observed_documents):
        fn()
        print(f"ok  {fn.__name__}")
