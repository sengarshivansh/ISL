import random

import numpy as np
import pytest
import torch

from islt.data import AugConfig, SignDataset, augment, nonmanual_tags, split_of, source_video
from islt.features import FACE_REGIONS, FEAT_DIM, N_FACE, face_points, make_streams, resample_speed
from islt.metrics import nonmanual_scores, translation_scores
from islt.stream import Segmenter, boundary_f1, segment_stream
from islt.synthetic import make_dataset, make_sentence, make_stream


@pytest.fixture(scope="module")
def toy():
    return make_dataset(16, seed=0)


def test_face_regions_cover_subset_without_overlap():
    base = ["mouth", "eyes", "eyebrows", "oval", "nose"]
    idx = [i for r in base for i in FACE_REGIONS[r]]
    assert sorted(idx) == list(range(N_FACE))
    assert face_points("none") == [] and len(face_points("all")) == N_FACE


def test_streams_shapes_and_invariance(toy):
    items, clips = toy
    raw = clips[items[0][0]]
    s = make_streams(raw)
    assert s["face"].shape == (len(raw), N_FACE * 2) and s["present"].shape[1] == 4
    # translating and scaling the whole person must not change body-normalised features
    moved = raw.copy()
    for start, stop, step in ((0, 132, 4), (132, FEAT_DIM, 3)):
        for k in range(start, stop, step):
            nz = moved[:, k] != 0
            moved[nz, k] = moved[nz, k] * 0.8 + 0.05
            moved[nz, k + 1] = moved[nz, k + 1] * 0.8 + 0.05
    s2 = make_streams(moved)
    for k in ("body", "hands", "face"):
        np.testing.assert_allclose(s[k], s2[k], atol=1e-3)


def test_missing_parts_are_safe():
    raw = np.zeros((10, FEAT_DIM), np.float32)
    s = make_streams(raw)
    assert all(np.isfinite(v).all() for v in s.values())


def test_resample_speed_lengths(toy):
    raw = toy[1][toy[0][0][0]]
    assert len(resample_speed(raw, 2.0)) == round(len(raw) / 2)
    assert len(resample_speed(raw, 0.5)) == len(raw) * 2


def test_split_keeps_videos_together():
    uids = [f"vid{i}-{j}" for i in range(200) for j in range(5)]
    by_video = {}
    for u in uids:
        by_video.setdefault(source_video(u), set()).add(split_of(u))
    assert all(len(s) == 1 for s in by_video.values())
    counts = [split_of(u) for u in uids]
    assert 0.8 < counts.count("train") / len(counts) < 0.97


def test_nonmanual_tags():
    assert nonmanual_tags("Are you coming?")["question"]
    assert nonmanual_tags("I don't know.")["negation"]
    assert not nonmanual_tags("Good morning")["negation"]


def test_augment_speed_changes_length(toy):
    raw = toy[1][toy[0][0][0]]
    out = augment(raw, AugConfig(speed_prob=1.0, speed_range=(2.0, 2.0)), random.Random(0))
    assert len(out) < len(raw)


def test_metrics_perfect_and_nonmanual():
    refs = ["you go home ?", "i not eat food", "we like school"]
    s = translation_scores(refs, refs)
    assert s["bleu4"] > 99 and s["chrf"] > 99
    nm = nonmanual_scores(["you go home", "i eat food", "we like school"], refs)
    assert nm["question_recall"] == 0 and nm["negation_recall"] == 0


def test_segmenter_finds_sentences():
    rng = random.Random(1)
    sents = [make_sentence(rng) for _ in range(4)]
    stream, _, gold = make_stream(sents, rng)
    segs = segment_stream(stream)
    assert boundary_f1(segs, gold, tol=5)["f1"] >= 75


def test_segmenter_decimates_30fps():
    seg = Segmenter()
    rng = random.Random(2)
    stream, _, _ = make_stream([make_sentence(rng)], rng)
    for i, f in enumerate(np.repeat(stream, 3, axis=0)):  # 30 fps input
        seg.push(f, t=i / 30)
    assert abs(seg.n - len(stream)) <= 2


@pytest.mark.parametrize("region,fusion", [("none", "gate"), ("all", "gate"), ("mouth", "gate"),
                                           ("all", "concat")])
def test_model_forward_and_generate(toy, region, fusion):
    from islt.data import make_collate
    from islt.model import FaSTISL, ModelConfig, load_tokenizer
    items, clips = toy
    tok = load_tokenizer()
    ds = SignDataset(items[:4], clips.__getitem__)
    batch = make_collate(tok)([ds[i] for i in range(4)])
    model = FaSTISL(ModelConfig(face_region=region, fusion=fusion, t5_name=None, d_model=64,
                                enc_layers=1))
    out = model(batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    gen = model.generate(batch, max_new_tokens=5, num_beams=1)
    assert gen.shape[0] == 4


def test_boundary_labels_and_training_stream():
    from islt.boundary import DELAY, labels_for, make_training_stream
    y = labels_for(50, [20], width=1)
    assert y[20 + DELAY] == 1 and y.sum() == 3
    items, clips = make_dataset(3, seed=4)
    stream, ends = make_training_stream([clips[u] for u, _ in items], random.Random(0))
    assert len(ends) == 2 and ends[-1] < len(stream)


def test_learned_segmenter_runs(tmp_path):
    from islt.boundary import BoundaryNet, LearnedSegmenter
    torch.save({"model": BoundaryNet().state_dict()}, tmp_path / "b.pt")
    seg = LearnedSegmenter(str(tmp_path / "b.pt"), threshold=0.0)  # untrained: must just not crash
    rng = random.Random(3)
    stream, _, _ = make_stream([make_sentence(rng) for _ in range(2)], rng)
    assert all(s.end > s.start for s in segment_stream(stream, segmenter=seg))


def test_trim_static_removes_holds():
    from islt.boundary import trim_static
    rng = random.Random(5)
    stream, _, _ = make_stream([make_sentence(rng)], rng)
    clip, _ = trim_static(stream, 0)
    assert clip is not None and len(clip) < len(stream)
    assert trim_static(np.repeat(stream[:1], 20, axis=0), 0)[0] is None


def test_word_dropout_and_contrastive_loss(toy):
    from islt.data import make_collate
    from islt.model import FaSTISL, ModelConfig, load_tokenizer
    items, clips = toy
    tok = load_tokenizer()
    ds = SignDataset(items[:4], clips.__getitem__)
    batch = make_collate(tok)([ds[i] for i in [0, 1, 2, 2]])  # a duplicate sentence: must count as positive
    model = FaSTISL(ModelConfig(t5_name=None, d_model=64, enc_layers=1, contrastive=0.3)).train()
    out = model(batch, word_dropout=0.5)
    assert torch.isfinite(out.loss) and torch.isfinite(out.aux_loss) and out.aux_loss > 0
    (out.loss + 0.3 * out.aux_loss).backward()
    assert model.con_sign.weight.grad is not None
    model.eval()
    assert float(model(batch).aux_loss) == 0.0  # contrastive loss is training-only


def test_paired_bootstrap_prefers_better_system(tmp_path):
    from islt.significance import paired_bootstrap
    refs = [f"the cat number {i} sat on the mat" for i in range(60)]
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    a.write_text("uid\treference\tprediction\n" + "".join(f"u{i}\t{r}\ta dog ran\n" for i, r in enumerate(refs)))
    b.write_text("uid\treference\tprediction\n" + "".join(f"u{i}\t{r}\t{r}\n" for i, r in enumerate(refs)))
    out = paired_bootstrap(a, b, n=50)
    assert out["bleu4"]["B"] > out["bleu4"]["A"] and out["bleu4"]["p_value"] < 0.05
    same = paired_bootstrap(a, a, n=50)
    assert same["bleu4"]["delta"] == 0 and same["bleu4"]["p_value"] == 1.0
