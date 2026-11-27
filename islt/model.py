"""
FaST-ISL model: Face-aware, Speed-robust pose-to-English translator.

    body, hands, present ──► manual projection ─┐
    face  ─► face proj ┐
    head pose ─► head proj ┴► f ──────────────┼─► gated fusion (h = m + g ⊙ f)
                                                ▼
             multi-rate temporal conv (dilations 1, 2, 4), stride 2
                                                ▼
                        Transformer encoder (pose tokens)
                                                ▼
              linear ──► T5 encoder (inputs_embeds) ──► T5 decoder ──► English

* Every stream is fed as [x, Δx] so motion is explicit (cf. Dhamecha et al., 2026).
* fusion="gate": per-channel sigmoid gate, so the model can switch the face
  off when it carries nothing (e.g. a frame without a non-manual marker).
  fusion="concat": plain concatenation (ablation). use_face=False: the baseline.
* The multi-rate temporal block sees the same motion at several time scales,
  which together with speed augmentation makes the encoder less sensitive to
  how fast the person signs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import STREAM_DIMS, face_points


@dataclass
class ModelConfig:
    face_region: str = "all"     # "all", "none", "mouth", "eyes", "eyebrows", "head", ...
    fusion: str = "gate"         # "gate" | "concat"
    d_model: int = 256
    enc_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    multirate: bool = True
    t5_name: str | None = "t5-small"   # pretrained decoder; None = small T5 from scratch
    freeze_t5_encoder: bool = False
    contrastive: float = 0.0     # weight of the clip<->sentence contrastive loss (0 = off)

    @property
    def use_face(self) -> bool:
        return self.face_region != "none"

    @property
    def face_dim(self) -> int:
        return len(face_points(self.face_region)) * 2

    def to_dict(self):
        return asdict(self)


def with_velocity(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(x)
    dx[:, 1:] = x[:, 1:] - x[:, :-1]
    dx = dx * (mask[:, :, None] & F.pad(mask[:, :-1], (1, 0))[:, :, None])
    return torch.cat([x, dx], dim=-1)


class Proj(nn.Module):
    def __init__(self, d_in, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class MultiRateConv(nn.Module):
    """Parallel depthwise temporal convs at dilations 1/2/4, then stride-2 downsampling."""

    def __init__(self, d, dilations=(1, 2, 4), multirate=True, dropout=0.1):
        super().__init__()
        dil = dilations if multirate else (1,)
        self.branches = nn.ModuleList(
            nn.Conv1d(d, d, 5, padding=2 * r, dilation=r, groups=d) for r in dil)
        self.mix = nn.Conv1d(d * len(dil), d, 1)
        self.down = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = (x * mask[..., None]).transpose(1, 2)
        h = self.mix(torch.cat([b(h) for b in self.branches], 1))
        h = F.gelu(h) + x.transpose(1, 2)
        h = self.down(h).transpose(1, 2)
        new_mask = mask[:, ::2]
        return self.drop(self.norm(h)), new_mask


def sinusoid(t, d, device):
    pos = torch.arange(t, device=device)[:, None]
    i = torch.arange(0, d, 2, device=device)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(t, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class PoseEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.body = Proj(STREAM_DIMS["body"] * 2, d, cfg.dropout)
        self.hands = Proj(STREAM_DIMS["hands"] * 2, d, cfg.dropout)
        self.present = nn.Linear(STREAM_DIMS["present"], d)
        if cfg.use_face:
            pts = torch.tensor(face_points(cfg.face_region))
            self.register_buffer("face_cols", torch.stack([2 * pts, 2 * pts + 1], 1).reshape(-1),
                                 persistent=False)
            self.face = Proj(cfg.face_dim * 2, d, cfg.dropout)
            self.head = Proj(STREAM_DIMS["head"] * 2, d, cfg.dropout)  # own stream: 6 numbers would drown in 300
            if cfg.fusion == "gate":
                self.gate = nn.Linear(2 * d, d)
                nn.init.constant_(self.gate.bias, -1.0)  # start mostly closed, open if useful
            else:
                self.fuse = nn.Linear(2 * d, d)
        self.temporal = MultiRateConv(d, multirate=cfg.multirate, dropout=cfg.dropout)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, cfg.enc_layers)
        self.last_gate: torch.Tensor | None = None

    def forward(self, batch):
        m = batch["mask"]
        manual = (self.body(with_velocity(batch["body"], m))
                  + self.hands(with_velocity(batch["hands"], m))
                  + self.present(batch["present"]))
        if self.cfg.use_face:
            face = batch["face"].index_select(-1, self.face_cols)
            f = self.face(with_velocity(face, m)) + self.head(with_velocity(batch["head"], m))
            if self.cfg.fusion == "gate":
                g = torch.sigmoid(self.gate(torch.cat([manual, f], -1)))
                self.last_gate = g.detach()
                h = manual + g * f
            else:
                h = self.fuse(torch.cat([manual, f], -1))
        else:
            h = manual
        h, m = self.temporal(h, m)
        h = h + sinusoid(h.shape[1], h.shape[2], h.device)
        h = self.transformer(h, src_key_padding_mask=~m)
        return h, m


class FaSTISL(nn.Module):
    def __init__(self, cfg: ModelConfig, t5=None):
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration
        self.cfg = cfg
        self.encoder = PoseEncoder(cfg)
        if t5 is not None:
            self.t5 = t5
        elif cfg.t5_name:
            self.t5 = T5ForConditionalGeneration.from_pretrained(cfg.t5_name, token=False)
        else:
            self.t5 = T5ForConditionalGeneration(T5Config(
                vocab_size=32128, d_model=256, d_kv=32, d_ff=1024, num_layers=3,
                num_decoder_layers=3, num_heads=8, dropout_rate=cfg.dropout,
                decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
        self.bridge = nn.Linear(cfg.d_model, self.t5.config.d_model)
        if cfg.contrastive > 0:
            dt = self.t5.config.d_model
            self.con_sign = nn.Linear(dt, 256)
            self.con_text = nn.Linear(dt, 256)
            self.con_logit_scale = nn.Parameter(torch.tensor(2.659))  # log(1/0.07), as in CLIP
        if cfg.freeze_t5_encoder:
            for p in self.t5.encoder.parameters():
                p.requires_grad = False

    def encode(self, batch):
        h, m = self.encoder(batch)
        return self.bridge(h), m.long()

    def forward(self, batch, word_dropout: float = 0.0):
        """T5 output; `.aux_loss` holds the contrastive loss (0 when off).

        word_dropout: during training, replace this share of the decoder's input words with <unk>,
        so the decoder cannot write the sentence from its language prior alone and must read the
        signing (the model otherwise drifts towards fluent, topic-level guesses).
        """
        emb, m = self.encode(batch)
        labels = batch["labels"]
        dec_in = self.t5._shift_right(labels)
        if self.training and word_dropout > 0:
            pad = self.t5.config.pad_token_id
            drop = (torch.rand(dec_in.shape, device=dec_in.device) < word_dropout) & (dec_in != pad)
            drop[:, 0] = False
            dec_in = dec_in.masked_fill(drop, 2)  # 2 = <unk> in the T5 vocabulary
        out = self.t5(inputs_embeds=emb, attention_mask=m, decoder_input_ids=dec_in, labels=labels)
        out.aux_loss = (self.contrastive_loss(out.encoder_last_hidden_state, m, labels)
                        if self.cfg.contrastive > 0 and self.training else labels.new_zeros((), dtype=torch.float))
        return out

    def contrastive_loss(self, sign_h, sign_mask, labels):
        """Symmetric InfoNCE between each clip and its own sentence (the other sentences in the
        batch are negatives). The sentence side is the same T5 encoder reading the target text."""
        def pool(h, mask):
            mask = mask.unsqueeze(-1).to(h.dtype)
            return (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        pad = self.t5.config.pad_token_id
        ids = labels.masked_fill(labels < 0, pad)
        tmask = (ids != pad).long()
        text_h = self.t5.encoder(input_ids=ids, attention_mask=tmask).last_hidden_state
        s = F.normalize(self.con_sign(pool(sign_h, sign_mask)), dim=-1)
        t = F.normalize(self.con_text(pool(text_h, tmask)), dim=-1)
        logits = self.con_logit_scale.exp().clamp(max=100) * s @ t.T
        # clips whose sentences are identical ("1", "Page number", ...) are all positives for each
        # other, not negatives: spread the target over every copy of the same sentence
        same = (ids[:, None, :] == ids[None, :, :]).all(-1).float()
        target = same / same.sum(1, keepdim=True)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2

    @torch.no_grad()
    def generate(self, batch, max_new_tokens=48, num_beams=4, no_repeat_ngram_size=3):
        emb, m = self.encode(batch)
        return self.t5.generate(inputs_embeds=emb, attention_mask=m, max_new_tokens=max_new_tokens,
                                num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)

    def n_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def count_encoder_params(cfg: ModelConfig) -> int:
    return sum(p.numel() for p in PoseEncoder(cfg).parameters())


def load_tokenizer(name: str = "t5-small"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, token=False)

