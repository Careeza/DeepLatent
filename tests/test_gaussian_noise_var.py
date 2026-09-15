"""Learned observation-noise variance for Gaussian (embedding) views."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from deeplatent.autoencoders import DecoderMLP, GaussianDecoderMLP
from deeplatent.corpus import Corpus
from deeplatent.models import IdealPointNN

LINEAR = {"hidden_dims": [], "activation": None, "bias": True}


def _ppca_data(n=4000, d=16, sigma2=0.25, seed=0):
    """One-factor PPCA: x = z w + b + eps, z ~ N(0, 1), eps ~ N(0, sigma2 I)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    w = rng.standard_normal(d)
    w *= 1.0 / np.linalg.norm(w)  # ||w||^2 = 1: top eigenvalue 1 + sigma2
    b = 0.5 * rng.standard_normal(d)
    x = np.outer(z, w) + b + math.sqrt(sigma2) * rng.standard_normal((n, d))
    return x.astype(np.float32), z


def _fit(x, tmp_path, decoder_extra, num_steps=2000, seed=0):
    corpus = Corpus(
        pd.DataFrame({"doc": [""] * len(x)}),
        modalities={
            "doc": {"views": {"embedding": {"type": "embedding", "matrix": x}}}
        },
    )
    model = IdealPointNN(
        corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=False,
        encoder_args={"doc_embedding": dict(LINEAR)},
        decoder_args={"doc_embedding": {**LINEAR, **decoder_extra}},
        num_steps=num_steps,
        batch_size=256,
        num_workers=0,
        optim_args={"main": {"lr": 5e-3}},
        print_every_n_steps=10**9,
        ckpt_folder=str(tmp_path / "ckpt"),
        return_best_model=False,
        device="cpu",
        seed=seed,
    )
    ip = model.get_ideal_points(corpus, num_samples=50).reshape(-1)
    return model, ip


def test_pinned_noise_var_matches_bare_squared_error_up_to_constant():
    """sigma^2 = 0.5 pinned is the old loss plus its dropped constant d/2 log(pi)."""
    torch.manual_seed(0)
    dec = GaussianDecoderMLP(decoder_dims=[2, 7], noise_var=0.5)
    recon, target = torch.randn(5, 3, 7), torch.randn(5, 1, 7)
    old = (recon - target).pow(2).sum(-1)
    new = dec.gaussian_nll(recon, target)
    assert new.shape == (5, 3)
    assert torch.allclose(new, old + 3.5 * math.log(math.pi), atol=1e-5)
    assert "noise_log_var" not in dict(dec.named_parameters())  # a buffer, not learned


def test_learned_noise_var_is_a_parameter_initialised_at_one():
    dec = GaussianDecoderMLP(decoder_dims=[2, 7], noise_var="learn")
    assert "noise_log_var" in dict(dec.named_parameters())
    assert float(dec.noise_log_var.exp()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        GaussianDecoderMLP(decoder_dims=[2, 7], noise_var=0.0)


def test_default_embedding_decoder_is_unchanged(tmp_path):
    """Without `noise_var` the view keeps the plain DecoderMLP and the old loss."""
    x, _ = _ppca_data(n=300)
    model, _ = _fit(x, tmp_path, {}, num_steps=5)
    dec = model.decoders["doc_embedding"]
    assert type(dec) is DecoderMLP and not hasattr(dec, "noise_log_var")


def test_learned_noise_var_recovers_ppca_and_is_scale_equivariant(tmp_path):
    """Linear decoder + Gaussian prior = PPCA: sigma^2 -> its ML value (the mean of
    the trailing eigenvalues) and the ideal point -> the first principal score."""
    sigma2 = 0.25
    x, _ = _ppca_data(sigma2=sigma2)
    xc = x - x.mean(0)
    pc1 = xc @ np.linalg.svd(xc, full_matrices=False)[2][0]
    model, ip = _fit(x, tmp_path, {"noise_var": "learn"})
    sigma2_hat = float(model.decoders["doc_embedding"].noise_log_var.detach().exp())
    assert sigma2_hat == pytest.approx(sigma2, rel=0.15)
    assert abs(np.corrcoef(ip, pc1)[0, 1]) > 0.98

    # x -> c x must rescale sigma^2 by c^2 and leave the ideal points unchanged.
    c = 3.0
    model_c, ip_c = _fit(c * x, tmp_path, {"noise_var": "learn"})
    sigma2_hat_c = float(model_c.decoders["doc_embedding"].noise_log_var.detach().exp())
    assert sigma2_hat_c / sigma2_hat == pytest.approx(c**2, rel=0.15)
    assert abs(np.corrcoef(ip_c, ip)[0, 1]) > 0.99


def test_checkpoint_roundtrip_keeps_noise_var(tmp_path):
    x, _ = _ppca_data(n=300)
    model, _ = _fit(x, tmp_path, {"noise_var": "learn"}, num_steps=20)
    with torch.no_grad():
        model.decoders["doc_embedding"].noise_log_var.fill_(-1.7)
    model.save_model(str(tmp_path / "m.ckpt"))
    other, _ = _fit(x, tmp_path / "b", {"noise_var": "learn"}, num_steps=2)
    other.load_model(str(tmp_path / "m.ckpt"))
    loaded = float(other.decoders["doc_embedding"].noise_log_var.detach())
    assert loaded == pytest.approx(-1.7)
