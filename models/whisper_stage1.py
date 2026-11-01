
import os
import json
import argparse
from collections import Counter
from typing import Optional, Tuple
import math

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import WhisperModel, WhisperFeatureExtractor


# ============================================================
# Final fixed configuration
# ============================================================

os.environ.pop("PL_TRAINER_AMP_BACKEND", None)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
TARGET_SEC = 30.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SEC)  # 480000 samples
MAX_MEL_FRAMES = 3000                           # 30 s -> 3000 mel frames

WHISPER_MODEL_NAME = "openai/whisper-small"
TRAIN_LAYER_START = 8
TRAIN_LAYER_END = 11
OUTPUT_LAYER = 11


# ============================================================
# Utilities
# ============================================================

def multilabel2vec(label_dict, label2idx):
    """Convert sparse emotion weights to a normalized dense vector."""
    vec = np.zeros(len(label2idx), dtype=np.float32)
    for emo, weight in label_dict.items():
        if emo in label2idx:
            vec[label2idx[emo]] = float(weight)

    total = vec.sum()
    if total > 0:
        vec /= total
    else:
        vec[:] = 1.0 / len(label2idx)
    return vec


def compute_valid_token_length(wav_len_samples: int) -> int:
    """
    Waveform samples -> valid Whisper encoder tokens.

    Whisper:
      mel frames ~= ceil(samples / 160)
      encoder tokens ~= ceil(mel_frames / 2)
    """
    if wav_len_samples <= 0:
        return 1

    n_mel_valid = int(math.ceil(wav_len_samples / float(HOP_LENGTH)))
    n_mel_valid = min(MAX_MEL_FRAMES, max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)


def build_token_mask(valid_wav_lens: torch.Tensor, T: int, device) -> torch.Tensor:
    """Build [B, T] mask: True=valid speech token, False=zero-padding token."""
    if not torch.is_tensor(valid_wav_lens):
        valid_wav_lens = torch.tensor(valid_wav_lens)

    valid_wav_lens = valid_wav_lens.detach().cpu().long().view(-1)
    valid_Ts = [
        min(compute_valid_token_length(int(wav_len.item())), T)
        for wav_len in valid_wav_lens
    ]
    valid_Ts = torch.tensor(valid_Ts, device=device, dtype=torch.long)
    t = torch.arange(T, device=device).view(1, T)
    return t < valid_Ts.view(-1, 1)


# ============================================================
# Whisper encoder
# ============================================================

class WhisperEncoderWrapper(nn.Module):
    """
    Final configuration:
      - Whisper-small
      - train encoder Layers 8-11
      - output encoder Layer 11
      - all other Whisper layers frozen
    """

    def __init__(self):
        super().__init__()

        self.model = WhisperModel.from_pretrained(WHISPER_MODEL_NAME)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            WHISPER_MODEL_NAME
        )
        self.hidden_size = self.model.config.d_model

        # Freeze all Whisper parameters first.
        for p in self.model.parameters():
            p.requires_grad = False

        # Human-readable Layers 8-11 correspond to Python indices 7:11.
        for layer in self.model.encoder.layers[TRAIN_LAYER_START - 1:TRAIN_LAYER_END]:
            for p in layer.parameters():
                p.requires_grad = True

        # Kept exactly as in the original experiment.
        for p in self.model.encoder.layer_norm.parameters():
            p.requires_grad = True

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        wav: torch.Tensor,
        valid_wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav_cpu = wav.detach().cpu().numpy()

        # The collate function already fixes every waveform to 30 s.
        # These options are retained as an additional safety check.
        feats = self.feature_extractor(
            wav_cpu,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=TARGET_SAMPLES,
        )

        device = next(self.model.parameters()).device
        input_features = feats.input_features.to(device)  # [B, 80, 3000]

        attention_mask = None
        if valid_wav_lens is not None:
            if len(valid_wav_lens) != input_features.size(0):
                raise RuntimeError(
                    f"valid_wav_lens mismatch: expected {input_features.size(0)}, "
                    f"got {len(valid_wav_lens)}"
                )

            token_T = (int(input_features.size(-1)) + 1) // 2
            attention_mask = build_token_mask(
                valid_wav_lens, token_T, input_features.device
            )

        # Compatibility fallback retained from the original implementation.
        try:
            encoder_out = self.model.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        except TypeError:
            encoder_out = self.model.encoder(
                input_features=input_features,
                return_dict=True,
                output_hidden_states=True,
            )

        hidden_states = encoder_out.hidden_states
        if OUTPUT_LAYER < 1 or OUTPUT_LAYER >= len(hidden_states):
            raise ValueError(
                f"OUTPUT_LAYER={OUTPUT_LAYER} is invalid; "
                f"hidden_states length={len(hidden_states)}"
            )

        # Fixed final output: Whisper encoder Layer 11.
        return hidden_states[OUTPUT_LAYER]


# ============================================================
# Dataset
# ============================================================

class PretrainEmoDataset(data.Dataset):
    """
    Returns:
      wav, hard_label, original_wav_len

    Collate:
      <30 s -> zero-pad at the end
      =30 s -> unchanged
      >30 s -> center-crop to 30 s
    """

    def __init__(self, datadir: str, labeldir: str, split: str = "Train"):
        with open(labeldir, "r", encoding="utf-8") as f:
            raw_label = json.load(f)

        self.raw_label = raw_label
        self.split = split

        if isinstance(raw_label, dict) and (
            "Train" in raw_label or "Val" in raw_label or "Test" in raw_label
        ):
            if split not in raw_label:
                raise ValueError(
                    f"label json missing split='{split}', keys={list(raw_label.keys())}"
                )
            self.label = raw_label[split]
            print(f"[PretrainEmoDataset] split='{split}', n={len(self.label)}")
        else:
            self.label = raw_label
            print(f"[PretrainEmoDataset] whole label dict, n={len(self.label)}")

        datadir = datadir.rstrip("/")
        self.datasetbase = [x for x in os.listdir(datadir) if x in self.label]
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]

        if len(self.label) == 0:
            raise ValueError(f"[PretrainEmoDataset] split='{self.split}' label empty")

        first_val = list(self.label.values())[0]
        if isinstance(first_val, str):
            self.emos = Counter(self.label.values())
        else:
            self.emos = Counter(
                emo
                for sparse_emo in self.label.values()
                for emo in sparse_emo.keys()
            )

        self.emoset = list(self.emos.keys())
        self.nemos = len(self.emoset)
        self.labeldict = {emo: idx for idx, emo in enumerate(self.emoset)}

        print("----Involved Emotions----")
        for emo, count in self.emos.items():
            print(f"{emo}: {count} examples")
        print(f"Total {len(self.dataset)} examples in split={self.split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        wav, _sr = sf.read(self.dataset[i])

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        label_data = self.label[self.datasetbase[i]]
        if isinstance(label_data, str):
            label_data = {label_data: 1.0}

        # Final configuration always uses a hard target.
        p = multilabel2vec(label_data, self.labeldict)
        label = int(np.argmax(p))

        wav = wav.astype(np.float32)
        wav_len = int(wav.shape[0])
        return wav, label, wav_len

    @staticmethod
    def _zero_pad_or_center_crop_to_length(
        x: np.ndarray,
        target_len: int,
    ) -> Tuple[np.ndarray, int]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        L = int(x.shape[0])

        # Empty audio.
        if L <= 0:
            return np.zeros((target_len,), dtype=np.float32), 0

        # >=30 s: center crop.
        if L >= target_len:
            start = (L - target_len) // 2
            seg = x[start:start + target_len]
            return seg.astype(np.float32, copy=False), target_len

        # <30 s: zero-pad at the end.
        out = np.zeros((target_len,), dtype=np.float32)
        out[:L] = x
        return out, L

    def pad30s_collate(self, batch):
        processed = []

        for wav, label, wav_len in batch:
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            wav_len = max(0, min(int(wav_len), wav.shape[0]))
            base = wav[:wav_len] if wav_len > 0 else wav

            wav_30s, valid_len = self._zero_pad_or_center_crop_to_length(
                base, TARGET_SAMPLES
            )
            processed.append((wav_30s, int(label), int(valid_len)))

        return default_collate(processed)


# ============================================================
# Heads
# ============================================================

class LinearHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class MaskedMeanPool(nn.Module):
    """Average valid time steps: [B, T, D] -> [B, D]. No learned parameters."""

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")

        # Accumulate in float32 to avoid overflow with mixed-precision training.
        x_float = x.float()
        if mask is None:
            pooled = x_float.mean(dim=1)
        else:
            if tuple(mask.shape) != tuple(x.shape[:2]):
                raise ValueError(
                    f"Mask shape {tuple(mask.shape)} does not match "
                    f"input shape {tuple(x.shape[:2])}"
                )
            mask = mask.to(device=x.device, dtype=torch.bool)
            valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            pooled = x_float.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
            pooled = pooled / valid_count

        return pooled.to(dtype=x.dtype)


# ============================================================
# Lightning model
# ============================================================

class WhisperLayer11EmoClassifier(pl.LightningModule):
    """
    Final Stage-1 configuration:
      Whisper Layers 8-11 trainable
      Whisper Layer 11 output
      downstream = masked temporal mean pooling + utterance MLP
      loss = utterance-level cross entropy only
      acc / valid_acc = utterance-level accuracy
    """

    def __init__(
        self,
        maxstep: int,
        batch_size: int,
        lr_head: float,
        lr_whisper: float,
        weight_decay: float,
        warmup_ratio: float,
        flat_ratio: float,
        min_lr_ratio: float,
        label_smoothing: float,
        head_dropout: float,
        datadir: str,
        labeldir: str,
        valid_split: float = 0.8,
        seed: int = 42,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.maxstep = int(maxstep)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_workers = int(num_workers)

        # Dataset and 80/20 train-validation split.
        self.data = PretrainEmoDataset(
            datadir=datadir,
            labeldir=labeldir,
            split="Train",
        )

        numtraining = int(len(self.data) * float(valid_split))
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(
            self.data,
            splits,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Compute class weights ONLY from the 80% optimization subset.
        # Match the hard-label conversion used in PretrainEmoDataset.__getitem__.
        train_class_counts = torch.zeros(self.data.nemos, dtype=torch.float)
        for idx in self.traindata.indices:
            filename = self.data.datasetbase[int(idx)]
            label_data = self.data.label[filename]
            if isinstance(label_data, str):
                label_data = {label_data: 1.0}
            label_vec = multilabel2vec(label_data, self.data.labeldict)
            label_id = int(np.argmax(label_vec))
            train_class_counts[label_id] += 1.0

        if (train_class_counts == 0).any():
            missing_classes = [
                emo for emo, count in zip(self.data.emoset, train_class_counts.tolist())
                if count == 0
            ]
            raise ValueError(
                f"Training subset contains no examples for classes {missing_classes}. "
                "Check the 80/20 split before computing inverse-frequency weights."
            )

        weights = train_class_counts.sum() / train_class_counts
        weights = weights / weights.sum()
        print(f"Training-subset class counts: {train_class_counts.tolist()}")
        print(f"Weigh losses by prior distribution of each class: {weights}.")

        self.num_classes = self.data.nemos
        self.register_buffer("class_weights", weights)

        try:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=float(self.hparams.label_smoothing),
            )
        except TypeError:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.whisper = WhisperEncoderWrapper()
        d_model = self.whisper.hidden_size

        self.utt_pool = MaskedMeanPool()
        self.utt_head = LinearHead(
            d_model, 128, self.num_classes, dropout=float(head_dropout)
        )

    def forward(
        self,
        wav_batch: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
    ):
        reps = self.whisper(wav_batch, valid_wav_lens=valid_len)

        token_mask = None
        if valid_len is not None:
            token_mask = build_token_mask(
                valid_len, reps.size(1), reps.device
            )

        utt_repr = self.utt_pool(reps, mask=token_mask)
        utt_logits = self.utt_head(utt_repr)
        return utt_logits

    def train_dataloader(self):
        nw = self.num_workers
        return data.DataLoader(
            self.traindata,
            batch_size=self.batch_size,
            num_workers=nw,
            drop_last=True,
            shuffle=True,
            collate_fn=self.data.pad30s_collate,
            persistent_workers=(nw > 0),
            pin_memory=True,
        )

    def val_dataloader(self):
        nw = self.num_workers
        return data.DataLoader(
            self.valdata,
            batch_size=self.batch_size,
            num_workers=nw,
            drop_last=False,
            shuffle=False,
            collate_fn=self.data.pad30s_collate,
            persistent_workers=(nw > 0),
            pin_memory=True,
        )

    def configure_optimizers(self):
        head_params = list(self.utt_head.parameters())
        whisper_params = self.whisper.trainable_params()

        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": float(self.hparams.lr_head)},
                {"params": whisper_params, "lr": float(self.hparams.lr_whisper)},
            ],
            weight_decay=float(self.hparams.weight_decay),
        )

        num_training_steps = int(self.maxstep)
        num_warmup_steps = int(
            float(self.hparams.warmup_ratio) * num_training_steps
        )
        num_flat_steps = int(
            float(self.hparams.flat_ratio) * num_training_steps
        )
        min_lr_ratio = float(self.hparams.min_lr_ratio)

        def lambda_lr(current_step: int):
            if num_training_steps <= 0:
                return 1.0

            if current_step < num_warmup_steps:
                return float(current_step + 1) / float(max(1, num_warmup_steps))

            if current_step < num_warmup_steps + num_flat_steps:
                return 1.0

            decay_steps = max(
                1,
                num_training_steps - (num_warmup_steps + num_flat_steps),
            )
            s = min(
                decay_steps,
                max(
                    0,
                    current_step - (num_warmup_steps + num_flat_steps),
                ),
            )
            progress = float(s) / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(optimizer, lambda_lr),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def _get_lrs_from_optimizer(opt):
        if opt is None or len(opt.param_groups) < 2:
            return None, None
        lr_head = float(opt.param_groups[0]["lr"])
        lr_whisper = float(opt.param_groups[1]["lr"])
        return lr_head, lr_whisper

    def training_step(self, batch, batch_idx):
        feats, label, valid_len = batch
        utt_logits = self(feats, valid_len)
        loss = self.criterion(utt_logits, label)
        acc = (utt_logits.argmax(dim=-1) == label).float().mean()

        lr_head, lr_whisper = None, None
        if (
            self.trainer is not None
            and hasattr(self.trainer, "optimizers")
            and len(self.trainer.optimizers) > 0
        ):
            lr_head, lr_whisper = self._get_lrs_from_optimizer(
                self.trainer.optimizers[0]
            )

        log_dict = {
            "loss": loss,
            "acc": acc,
        }
        if lr_head is not None:
            log_dict["lr_head"] = lr_head
        if lr_whisper is not None:
            log_dict["lr_whisper"] = lr_whisper

        self.log_dict(
            log_dict,
            batch_size=label.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feats, label, valid_len = batch
        utt_logits = self(feats, valid_len)
        loss = self.criterion(utt_logits, label)
        acc = (utt_logits.argmax(dim=-1) == label).float().mean()

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_acc": acc,
            },
            batch_size=label.size(0),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Paths.
    parser.add_argument(
        "--saving_path",
        type=str,
        default="./outputs/stage1_sess1",
)
    parser.add_argument(
    "--datadir",
    type=str,
    default="./dataset/Audio_16k",
)

    parser.add_argument(
    "--labelpath",
    type=str,
    default="./dataset/labels_sess/label_1.json",
)

    # Optimizer / scheduler.
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_whisper", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--flat_ratio", type=float, default=0.20)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.10)

    # Utterance-level MLP (no frame head or auxiliary frame loss).
    parser.add_argument("--head_dropout", type=float, default=0.5)

    # Trainer.
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--precision", type=int, choices=[16, 32], default=16)
    parser.add_argument("--training_step", type=int, default=15000)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    # Split / reproducibility.
    parser.add_argument(
        "--valid_split",
        type=float,
        default=0.8,
        help="Training proportion: 0.8 means 80% train / 20% validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)

    # Optional CSV metrics logging.
    parser.add_argument(
        "--enable_csv_logger",
        action="store_true",
        help="Write metrics.csv under saving_path/csv_logs/.",
    )

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.saving_path,
        filename=(
            "whisper-s1-meanmlp-layer11-out11-"
            "{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}"
        ),
        save_top_k=args.save_top_k,
        verbose=True,
        monitor="valid_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=10,
        verbose=True,
    )

    pl_precision = "16-mixed" if args.precision == 16 else "32-true"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    logger = False
    enable_progress_bar = True
    if args.enable_csv_logger:
        logger = CSVLogger(save_dir=args.saving_path, name="csv_logs")
        enable_progress_bar = False

    trainer = Trainer(
        precision=pl_precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_steps=args.training_step,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )

    model = WhisperLayer11EmoClassifier(
        maxstep=args.training_step,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_whisper=args.lr_whisper,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        flat_ratio=args.flat_ratio,
        min_lr_ratio=args.min_lr_ratio,
        label_smoothing=args.label_smoothing,
        head_dropout=args.head_dropout,
        datadir=args.datadir,
        labeldir=args.labelpath,
        valid_split=args.valid_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    ckpt_path = (
        args.resume_checkpoint
        if args.resume_checkpoint not in [None, "None", ""]
        else None
    )
    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
