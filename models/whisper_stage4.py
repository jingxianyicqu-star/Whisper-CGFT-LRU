
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math
import json
import inspect
from collections import Counter
from contextlib import nullcontext
from typing import Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import random_split
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from transformers import WhisperModel, WhisperFeatureExtractor


os.environ.pop("PL_TRAINER_AMP_BACKEND", None)


# ============================================================
# Session paths: relative to the repository root
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_ID = 1

# build_saving_path() appends a per-session / per-seed subdirectory.
BASE_SAVING_DIR = os.path.join(
    PROJECT_ROOT, "outputs", f"stage4_sess{SESSION_ID}"
)
DATADIR = os.path.join(PROJECT_ROOT, "dataset", "Audio_16k")
LABELPATH = os.path.join(
    PROJECT_ROOT, "dataset", "labels_sess", f"label_{SESSION_ID}.json"
)

# Replace this with the Stage3 checkpoint actually used for this fold.
# Loader supports both:
#   new Stage3: whisper.encoder....
#   old Stage3: whisper.model.encoder....
STAGE3_CKPT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    f"stage3_sess{SESSION_ID}",
    "whisper-s3-layer11-epoch=28-val_loss=5.6432-v2.ckpt",
)

RESUME_CHECKPOINT = None


# ============================================================
# Fixed final configuration
# ============================================================

WHISPER_MODEL_NAME = "openai/whisper-small"

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
TARGET_SEC = 30.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SEC)  # 480000
T_ENC = 1500

OUTPUT_LAYER = 11
TRAIN_LAYER_START = 8
TRAIN_LAYER_END = 11

VALID_SPLIT = 0.8
SEED = 42

LRU_LAYERS = 2
LRU_D_STATE = 128
LRU_DROPOUT = 0.15

# AvePooling+MLP classifier.
# Kept consistent with the Stage1 mean-pooling + MLP head.
MLP_HIDDEN_DIM = 128
MLP_DROPOUT = 0.50

LABEL_SMOOTHING = 0.05

LR_ENC = 5e-6
LR_HEAD = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.10
FLAT_RATIO = 0.10

BATCH_SIZE = 16
NUM_WORKERS = 4
PRECISION = 16
TRAINING_STEP = 10000
SAVE_TOP_K = 1
CHECK_VAL_EVERY_N_EPOCH = 1
EARLY_STOP_PATIENCE = 12
ACCUMULATE_GRAD_BATCHES = 2
GRAD_CLIP = 1.0

ENABLE_CSV_LOGGER = False


# ============================================================
# Utilities
# ============================================================

def multilabel2vec(label_dict: Dict[str, float], label2idx: Dict[str, int]) -> np.ndarray:
    """Sparse emotion weights -> normalized dense vector."""
    vec = np.zeros(len(label2idx), dtype=np.float32)
    for emotion, weight in label_dict.items():
        if emotion in label2idx:
            vec[label2idx[emotion]] = float(weight)

    total = float(vec.sum())
    if total > 0:
        vec /= total
    else:
        vec[:] = 1.0 / max(1, len(label2idx))
    return vec


def compute_valid_token_length(wav_len_samples: int) -> int:
    """Valid waveform samples -> valid Whisper encoder-token length."""
    if wav_len_samples <= 0:
        return 1

    n_mel = int(math.ceil(wav_len_samples / float(HOP_LENGTH)))
    n_mel = min(MAX_MEL_FRAMES, max(1, n_mel))
    return max(1, (n_mel + 1) // 2)


def build_valid_token_mask(
    valid_wav_lens: torch.Tensor,
    token_length: int,
    device,
) -> torch.Tensor:
    """Return [B,T] bool mask. True=real speech, False=zero padding."""
    if not torch.is_tensor(valid_wav_lens):
        valid_wav_lens = torch.tensor(valid_wav_lens)

    valid_wav_lens = valid_wav_lens.detach().cpu().long().view(-1)
    valid_Ts = [
        min(compute_valid_token_length(int(x.item())), token_length)
        for x in valid_wav_lens
    ]
    valid_Ts = torch.tensor(valid_Ts, device=device, dtype=torch.long)
    t = torch.arange(token_length, device=device).view(1, token_length)
    return t < valid_Ts.view(-1, 1)


@torch.no_grad()
def wa_ua_from_preds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate WA (overall accuracy) and UA (macro class recall).
    UA is the same metric often called UAR in SER papers.
    Only classes present in the evaluated split contribute to UA,
    preserving the metric convention of the original experiment.
    Confusion-matrix bincount is intentionally done on CPU.
    """
    device = preds.device
    K = int(num_classes)

    preds_cpu = preds.detach().long().cpu()
    targets_cpu = targets.detach().long().cpu()

    indices = targets_cpu * K + preds_cpu
    confusion = torch.bincount(indices, minlength=K * K).reshape(K, K)

    wa = torch.diag(confusion).sum().float() / confusion.sum().float().clamp_min(1.0)

    class_count = confusion.sum(dim=1).float()
    recall = torch.diag(confusion).float() / class_count.clamp_min(eps)
    present = class_count > 0
    ua = recall[present].mean() if present.any() else torch.tensor(0.0)

    return wa.to(device), ua.to(device), confusion.to(device)


# ============================================================
# Dataset
# ============================================================

class PretrainEmoDataset(data.Dataset):
    """
    Final Stage4 dataset.

    Audio:
        <30 s -> zero-pad at the end
        =30 s -> unchanged
        >30 s -> center-crop

    Output:
        waveform, hard emotion label, original waveform length
    """

    def __init__(
        self,
        datadir: str,
        labeldir: str,
        split: str,
        labeldict: Optional[Dict[str, int]] = None,
        emoset=None,
    ):
        with open(labeldir, "r", encoding="utf-8") as f:
            raw_label = json.load(f)

        if isinstance(raw_label, dict) and any(
            key in raw_label for key in ["Train", "Val", "Test"]
        ):
            if split not in raw_label:
                raise ValueError(
                    f"label json missing split='{split}', keys={list(raw_label.keys())}"
                )
            self.label = raw_label[split]
        else:
            self.label = raw_label

        self.split = split
        datadir = datadir.rstrip("/")

        self.datasetbase = [
            name for name in os.listdir(datadir) if name in self.label
        ]
        self.dataset = [
            os.path.join(datadir, name) for name in self.datasetbase
        ]

        if len(self.dataset) == 0:
            raise ValueError(
                f"[Dataset] 0 matching wav files: datadir={datadir}, split={split}"
            )

        raw_emotions = []
        for value in self.label.values():
            if isinstance(value, str):
                raw_emotions.append(value)
            else:
                raw_emotions.extend(value.keys())

        self.emos = Counter(raw_emotions)

        if labeldict is None or emoset is None:
            self.emoset = list(self.emos.keys())
            self.labeldict = {
                emotion: idx for idx, emotion in enumerate(self.emoset)
            }
        else:
            self.emoset = list(emoset)
            self.labeldict = dict(labeldict)

            unknown = sorted(set(raw_emotions) - set(self.labeldict.keys()))
            if unknown:
                raise ValueError(
                    f"[Dataset] split={split} contains unknown emotions {unknown}; "
                    f"train mapping={self.emoset}"
                )

        self.nemos = len(self.emoset)

        print(f"[PretrainEmoDataset] split='{split}', n={len(self.dataset)}")
        print("----Involved Emotions----")
        for emotion, count in self.emos.items():
            print(f"{emotion}: {count} examples")

    def __len__(self):
        return len(self.dataset)

    def _hard_label_from_raw(self, value) -> int:
        if isinstance(value, str):
            if value not in self.labeldict:
                raise ValueError(f"Unknown emotion label: {value}")
            return int(self.labeldict[value])

        probability = multilabel2vec(value, self.labeldict)
        return int(np.argmax(probability))

    def hard_label_at_index(self, index: int) -> int:
        """Get a label without reading the waveform."""
        name = self.datasetbase[int(index)]
        return self._hard_label_from_raw(self.label[name])

    def __getitem__(self, index):
        wav, sr = sf.read(self.dataset[index])

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"[Dataset] sample rate={sr}, expected={SAMPLE_RATE} "
                f"@ {self.datasetbase[index]}"
            )

        label = self._hard_label_from_raw(
            self.label[self.datasetbase[index]]
        )

        wav = wav.astype(np.float32)
        return wav, label, int(wav.shape[0])

    @staticmethod
    def _zero_pad_or_center_crop(wav: np.ndarray) -> Tuple[np.ndarray, int]:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        length = int(wav.shape[0])

        if length <= 0:
            return np.zeros(TARGET_SAMPLES, dtype=np.float32), 0

        if length >= TARGET_SAMPLES:
            start = (length - TARGET_SAMPLES) // 2
            segment = wav[start:start + TARGET_SAMPLES]
            return segment.astype(np.float32, copy=False), TARGET_SAMPLES

        output = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        output[:length] = wav
        return output, length

    def pad30s_collate(self, batch):
        processed = []

        for wav, label, wav_len in batch:
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            wav_len = max(0, min(int(wav_len), wav.shape[0]))
            base = wav[:wav_len] if wav_len > 0 else wav

            wav_30s, valid_len = self._zero_pad_or_center_crop(base)
            processed.append((wav_30s, int(label), int(valid_len)))

        return default_collate(processed)


# ============================================================
# Masked temporal mean pooling (no trainable parameters)
# ============================================================

class MaskedMeanPool(nn.Module):
    """Average valid time steps only: [B, T, D] -> [B, D]."""

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")
        if tuple(valid_mask.shape) != tuple(x.shape[:2]):
            raise ValueError(
                f"Mask shape {tuple(valid_mask.shape)} does not match "
                f"input shape {tuple(x.shape[:2])}"
            )
        valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
        # Float32 accumulation avoids overflow during mixed-precision training.
        x_float = x.float().masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        return (x_float.sum(dim=1) / count).to(dtype=x.dtype)


class MLPHead(nn.Module):
    """Utterance-level MLP used after temporal average pooling."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
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


# ============================================================
# LRU
# ============================================================

class LRULayer(nn.Module):
    """
    Preserve the original LRU recurrence:
        s_t = lambda * s_(t-1) + u_t
    """

    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        self.dropout = nn.Dropout(dropout)

        self.D = nn.Parameter(torch.ones(d_model))
        self.nu = nn.Parameter(torch.randn(d_state) * 0.1)
        self.theta = nn.Parameter(torch.rand(d_state) * (2.0 * math.pi))

    def _lambda(self, device):
        magnitude = torch.exp(-torch.exp(self.nu)).to(device)
        theta = self.theta.to(device)
        lam = magnitude * (torch.cos(theta) + 1j * torch.sin(theta))
        return lam.to(torch.complex64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Complex recurrence is intentionally outside fp16 autocast.
        ctx = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()

        with ctx:
            x_f = x.float()
            B, T, _ = x_f.shape

            u = self.in_proj(x_f).to(torch.complex64)
            lam = self._lambda(x_f.device)

            state = torch.zeros(
                B, self.d_state,
                device=x_f.device,
                dtype=torch.complex64,
            )
            states = torch.empty(
                B, T, self.d_state,
                device=x_f.device,
                dtype=torch.complex64,
            )

            for t in range(T):
                state = lam * state + u[:, t, :]
                states[:, t, :] = state

            y = self.out_proj(states.real)
            y = y + x_f * self.D
            y = F.gelu(y)
            return self.dropout(y)


class LRUBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lru = LRULayer(d_model, d_state, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lru(self.norm(x))


class LRUStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            LRUBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================
# Whisper Layer-11 wrapper
# ============================================================

class WhisperLayer11Wrapper(nn.Module):
    """
    Final Stage4 Whisper behavior:
        - create Whisper-small
        - load Whisper weights from Stage3 checkpoint
        - freeze all Whisper parameters
        - unfreeze encoder Layers 8-11
        - execute encoder only through Layer 11
        - return Layer-11 representations

    The encoder's final LayerNorm remains frozen, matching the uploaded
    Stage4 code's trainability behavior.
    """

    def __init__(self, stage3_ckpt_path: str):
        super().__init__()

        self.model = WhisperModel.from_pretrained(WHISPER_MODEL_NAME)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            WHISPER_MODEL_NAME
        )
        self.hidden_size = int(self.model.config.d_model)
        self._fallback_warned = False

        self._load_stage3_whisper(stage3_ckpt_path)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for layer in self.model.encoder.layers[
            TRAIN_LAYER_START - 1:TRAIN_LAYER_END
        ]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        print(
            f"[Stage4] Whisper train layers = "
            f"{TRAIN_LAYER_START}-{TRAIN_LAYER_END}"
        )
        print(f"[Stage4] Whisper output layer = {OUTPUT_LAYER}")

    def _load_stage3_whisper(self, ckpt_path: str):
        """
        Load Whisper parameters only.

        Supported checkpoint formats:
          new Stage3 : whisper.encoder....
          old Stage3 : whisper.model.encoder....
        """
        if ckpt_path in [None, "", "None", "none"]:
            raise ValueError("[Stage4] STAGE3_CKPT_PATH is empty.")

        ckpt_path = os.path.expanduser(str(ckpt_path))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"[Stage4] Stage3 checkpoint not found: {ckpt_path}"
            )

        print(f"[Stage4] Loading Whisper from Stage3:\n{ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        if not isinstance(state_dict, dict):
            raise RuntimeError("[Stage4] Invalid Stage3 state_dict.")

        target_state = self.model.state_dict()

        prefixes = [
            "whisper.",
            "whisper.model.",
            "module.whisper.",
            "module.whisper.model.",
        ]

        best_state = {}
        best_prefix = None

        for prefix in prefixes:
            candidate = {}
            for key, value in state_dict.items():
                if not key.startswith(prefix):
                    continue

                stripped = key[len(prefix):]
                if (
                    stripped in target_state
                    and hasattr(value, "shape")
                    and value.shape == target_state[stripped].shape
                ):
                    candidate[stripped] = value

            if len(candidate) > len(best_state):
                best_state = candidate
                best_prefix = prefix

        if len(best_state) == 0:
            sample_keys = list(state_dict.keys())[:20]
            raise RuntimeError(
                "[Stage4] No compatible Whisper tensors found in Stage3 ckpt.\n"
                f"Sample keys: {sample_keys}"
            )

        missing, unexpected = self.model.load_state_dict(
            best_state,
            strict=False,
        )

        print(
            f"[Stage4] Stage3 Whisper load: prefix='{best_prefix}', "
            f"matched={len(best_state)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    def trainable_params(self):
        return [
            p for p in self.parameters()
            if p.requires_grad
        ]

    @staticmethod
    def _encoder_embed(
        encoder,
        input_features: torch.Tensor,
    ) -> torch.Tensor:
        x = F.gelu(encoder.conv1(input_features))
        x = F.gelu(encoder.conv2(x))
        x = x.permute(0, 2, 1)

        positional = None
        if hasattr(encoder, "embed_positions") and encoder.embed_positions is not None:
            try:
                positional = encoder.embed_positions(x.shape)
            except Exception:
                try:
                    positional = encoder.embed_positions.weight[:x.size(1)]
                except Exception:
                    positional = None

        if positional is not None:
            if positional.dim() == 2:
                positional = positional.unsqueeze(0)
            x = x + positional.to(device=x.device, dtype=x.dtype)

        dropout = getattr(encoder, "dropout", 0.0)
        if isinstance(dropout, nn.Dropout):
            x = dropout(x)
        else:
            try:
                p = float(dropout)
                if p > 0:
                    x = F.dropout(x, p=p, training=encoder.training)
            except Exception:
                pass

        return x

    @staticmethod
    def _expand_attention_mask(
        valid_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """[B,T] valid mask -> [B,1,T,T] additive mask."""
        valid_mask = valid_mask.bool()
        B, T = valid_mask.shape

        additive = (~valid_mask).to(dtype=dtype) * torch.finfo(dtype).min
        return additive[:, None, None, :].expand(B, 1, T, T)

    @staticmethod
    def _call_encoder_layer(
        layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Small compatibility layer for nearby transformers versions."""
        signature = inspect.signature(layer.forward)
        kwargs = {}

        if "attention_mask" in signature.parameters:
            kwargs["attention_mask"] = attention_mask
        if "layer_head_mask" in signature.parameters:
            kwargs["layer_head_mask"] = None
        if "output_attentions" in signature.parameters:
            kwargs["output_attentions"] = False

        output = layer(hidden_states, **kwargs)
        return output[0] if isinstance(output, (tuple, list)) else output

    def encode_layer11(
        self,
        wav_30s: torch.Tensor,
        valid_wav_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          reps       [B,T,C], Whisper encoder Layer 11
          valid_mask [B,T], True for real speech tokens
        """
        if wav_30s.dim() == 1:
            wav_30s = wav_30s.unsqueeze(0)

        wav_cpu = wav_30s.detach().cpu().numpy()

        input_features = self.feature_extractor(
            wav_cpu,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=TARGET_SAMPLES,
        ).input_features

        device = next(self.model.parameters()).device
        input_features = input_features.to(device)

        try:
            encoder = self.model.encoder
            hidden_states = self._encoder_embed(encoder, input_features)
            T = int(hidden_states.size(1))

            valid_mask = build_valid_token_mask(
                valid_wav_lens,
                token_length=T,
                device=hidden_states.device,
            )
            attention_mask = self._expand_attention_mask(
                valid_mask,
                hidden_states.dtype,
            )

            # Layer index 0..10 => human-readable Layers 1..11.
            for layer_index in range(OUTPUT_LAYER):
                hidden_states = self._call_encoder_layer(
                    encoder.layers[layer_index],
                    hidden_states,
                    attention_mask,
                )

            return hidden_states, valid_mask

        except Exception as exc:
            if not self._fallback_warned:
                print(
                    "[WhisperLayer11Wrapper][WARN] manual Layer-11 path "
                    "failed; fallback to output_hidden_states=True."
                )
                print(f"[WhisperLayer11Wrapper][WARN] reason: {repr(exc)}")
                self._fallback_warned = True

            valid_mask = build_valid_token_mask(
                valid_wav_lens,
                token_length=T_ENC,
                device=input_features.device,
            )

            output = self.model.encoder(
                input_features=input_features,
                attention_mask=valid_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            reps = output.hidden_states[OUTPUT_LAYER]

            if valid_mask.size(1) != reps.size(1):
                valid_mask = build_valid_token_mask(
                    valid_wav_lens,
                    token_length=int(reps.size(1)),
                    device=reps.device,
                )

            return reps, valid_mask


# ============================================================
# Stage4 Lightning model
# ============================================================

class WhisperLRUDownstreamClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.maxstep = TRAINING_STEP
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS

        # ----------------------------------------------------
        # Train / validation: fixed 80/20
        # ----------------------------------------------------
        train_base = PretrainEmoDataset(
            DATADIR,
            LABELPATH,
            split="Train",
        )

        n_total = len(train_base)
        if n_total < 2:
            raise ValueError(f"[Stage4] Train split too small: N={n_total}")

        n_train = int(round(n_total * VALID_SPLIT))
        n_train = min(max(1, n_train), n_total - 1)
        n_val = n_total - n_train

        generator = torch.Generator().manual_seed(SEED)
        self.traindata, self.valdata = random_split(
            train_base,
            [n_train, n_val],
            generator=generator,
        )

        self._train_collate_fn = train_base.pad30s_collate
        self._val_collate_fn = train_base.pad30s_collate

        print("[Stage4] Split strategy = random Train split 80/20.")
        print(f"[Stage4] Train={n_train}, Val={n_val}, seed={SEED}")

        # Test uses exactly the same label-index mapping as Train.
        self.testdata = PretrainEmoDataset(
            DATADIR,
            LABELPATH,
            split="Test",
            labeldict=train_base.labeldict,
            emoset=train_base.emoset,
        )
        self.num_classes = train_base.nemos

        # ----------------------------------------------------
        # Class weights: actual 80% optimization subset
        # ----------------------------------------------------
        class_counts = [0] * self.num_classes
        for subset_index in self.traindata.indices:
            label_index = train_base.hard_label_at_index(subset_index)
            class_counts[label_index] += 1

        class_count_dict = dict(zip(train_base.emoset, class_counts))
        print(f"[Stage4] Train subset class counts: {class_count_dict}")

        count_tensor = torch.tensor(
            [max(1, count) for count in class_counts],
            dtype=torch.float,
        )
        weights = count_tensor.sum() / count_tensor
        weights = weights / weights.sum()

        print(f"[Stage4] Class weights: {weights}")
        self.register_buffer("class_weights", weights)

        try:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=LABEL_SMOOTHING,
            )
        except TypeError:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights
            )

        # ----------------------------------------------------
        # Stage3-adapted Whisper + final LRU downstream
        # ----------------------------------------------------
        self.whisper = WhisperLayer11Wrapper(STAGE3_CKPT_PATH)
        d_model = int(self.whisper.hidden_size)

        self.lru = LRUStack(
            d_model=d_model,
            n_layers=LRU_LAYERS,
            d_state=LRU_D_STATE,
            dropout=LRU_DROPOUT,
        )
        self.utt_pool = MaskedMeanPool()
        self.utt_head = MLPHead(
            in_dim=d_model,
            hidden_dim=MLP_HIDDEN_DIM,
            num_classes=self.num_classes,
            dropout=MLP_DROPOUT,
        )

        self.val_step_outputs = []
        self.test_step_outputs = []

        print(f"[Stage4] Encoder token length T_enc={T_ENC}")

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        wav_batch: torch.Tensor,
        valid_len_samples: torch.Tensor,
    ):
        reps, valid_mask = self.whisper.encode_layer11(
            wav_batch,
            valid_len_samples,
        )

        reps = reps.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        seq = self.lru(reps)
        utt_repr = self.utt_pool(seq, valid_mask)
        utt_logits = self.utt_head(utt_repr)

        return utt_logits

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    def train_dataloader(self):
        return data.DataLoader(
            self.traindata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self._train_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.valdata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self._val_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.testdata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.testdata.pad30s_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    def configure_optimizers(self):
        encoder_params = list(self.whisper.trainable_params())
        downstream_params = (
            list(self.lru.parameters())
            + list(self.utt_pool.parameters())
            + list(self.utt_head.parameters())
        )

        optimizer = optim.AdamW(
            [
                {"params": encoder_params, "lr": LR_ENC},
                {"params": downstream_params, "lr": LR_HEAD},
            ],
            weight_decay=WEIGHT_DECAY,
        )

        num_warmup = int(WARMUP_RATIO * self.maxstep)
        num_flat = int(FLAT_RATIO * self.maxstep)

        def lambda_lr(step: int):
            if step < num_warmup:
                return float(step) / float(max(1, num_warmup))

            if step < num_warmup + num_flat:
                return 1.0

            return max(
                0.0,
                float(self.maxstep - step)
                / float(max(1, self.maxstep - (num_warmup + num_flat))),
            )

        scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda_lr,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    # --------------------------------------------------------
    # Train / validation / test
    # --------------------------------------------------------

    def training_step(self, batch, batch_idx):
        wav_batch, labels, valid_len = batch
        wav_batch = wav_batch.to(self.device).float()
        labels = labels.to(self.device).long()
        valid_len = valid_len.to(self.device).long()

        utt_logits = self(
            wav_batch,
            valid_len,
        )

        loss = self.criterion(utt_logits, labels)

        prediction = torch.argmax(utt_logits, dim=-1)
        acc_utt = (prediction == labels).float().mean()

        self.log_dict(
            {
                "loss": loss,
                "loss_utt": loss,
                "acc_utt": acc_utt,
            },
            batch_size=labels.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        wav_batch, labels, valid_len = batch
        wav_batch = wav_batch.to(self.device).float()
        labels = labels.to(self.device).long()
        valid_len = valid_len.to(self.device).long()

        utt_logits = self(
            wav_batch,
            valid_len,
        )

        loss = self.criterion(utt_logits, labels)

        prediction = torch.argmax(utt_logits, dim=-1)

        self.log(
            "valid_loss",
            loss,
            batch_size=labels.size(0),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.val_step_outputs.append({
            "pred": prediction.detach().cpu(),
            "target": labels.detach().cpu(),
        })
        return loss

    def test_step(self, batch, batch_idx):
        wav_batch, labels, valid_len = batch
        wav_batch = wav_batch.to(self.device).float()
        labels = labels.to(self.device).long()
        valid_len = valid_len.to(self.device).long()

        utt_logits = self(
            wav_batch,
            valid_len,
        )

        loss = self.criterion(utt_logits, labels)

        prediction = torch.argmax(utt_logits, dim=-1)

        self.log("test_loss", loss, batch_size=labels.size(0), on_epoch=True, logger=True)

        self.test_step_outputs.append({
            "pred": prediction.detach().cpu(),
            "target": labels.detach().cpu(),
        })
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_step_outputs) == 0:
            return

        predictions = torch.cat(
            [x["pred"] for x in self.val_step_outputs],
            dim=0,
        ).to(self.device)
        targets = torch.cat(
            [x["target"] for x in self.val_step_outputs],
            dim=0,
        ).to(self.device)

        wa, ua, _ = wa_ua_from_preds(
            predictions,
            targets,
            self.num_classes,
        )

        self.log(
            "valid_WA",
            wa,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_UA",
            ua,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if not getattr(self.trainer, "sanity_checking", False):
            print(f"[Validation] WA = {wa.item() * 100:.2f}%, UA = {ua.item() * 100:.2f}%")

        self.val_step_outputs.clear()

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        predictions = torch.cat(
            [x["pred"] for x in self.test_step_outputs],
            dim=0,
        ).to(self.device)
        targets = torch.cat(
            [x["target"] for x in self.test_step_outputs],
            dim=0,
        ).to(self.device)

        wa, ua, _ = wa_ua_from_preds(
            predictions,
            targets,
            self.num_classes,
        )

        self.log(
            "test_WA",
            wa,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_UA",
            ua,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        print(f"[Test] WA = {wa.item() * 100:.2f}%, UA = {ua.item() * 100:.2f}%")
        self.test_step_outputs.clear()


# ============================================================
# Runner
# ============================================================

def build_saving_path() -> str:
    return os.path.join(
        os.path.expanduser(BASE_SAVING_DIR),
        f"sess{SESSION_ID}_layer{OUTPUT_LAYER}_lru_avepool_mlp_seed{SEED}",
    )


def main():
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    saving_path = build_saving_path()
    os.makedirs(saving_path, exist_ok=True)

    print(f"[Stage4] saving_path = {saving_path}")
    print("[Stage4] ========================================")
    print(f"[Stage4] Session = {SESSION_ID}")
    print(f"[Stage4] Whisper = {WHISPER_MODEL_NAME}")
    print(
        f"[Stage4] Train Layers = "
        f"{TRAIN_LAYER_START}-{TRAIN_LAYER_END}"
    )
    print(f"[Stage4] Output Layer = {OUTPUT_LAYER}")
    print("[Stage4] Audio = <30s zero-pad, >30s center-crop")
    print("[Stage4] Train/Val = 80/20")
    print("[Stage4] Downstream = LRU -> AvePooling -> MLP")
    print("[Stage4] Loss = utterance-level cross entropy only")
    print(
        f"[Stage4] LRU = {LRU_LAYERS} layers, "
        f"d_state={LRU_D_STATE}"
    )
    print(
        f"[Stage4] MLP = {MLP_HIDDEN_DIM} hidden units, "
        f"dropout={MLP_DROPOUT}"
    )
    print("[Stage4] ========================================")

    model = WhisperLRUDownstreamClassifier()

    checkpoint_callback = ModelCheckpoint(
        dirpath=saving_path,
        filename=(
            f"whisper-s4-layer{OUTPUT_LAYER}-lru-avemlp-sess{SESSION_ID}-"
            "{epoch:02d}-valid_loss={valid_loss:.4f}-"
            "valid_UA={valid_UA:.4f}-valid_WA={valid_WA:.4f}"
        ),
        save_top_k=SAVE_TOP_K,
        monitor="valid_loss",
        mode="min",
        verbose=True,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=EARLY_STOP_PATIENCE,
        verbose=True,
    )

    logger = False
    enable_progress_bar = True

    if ENABLE_CSV_LOGGER:
        logger = CSVLogger(
            save_dir=saving_path,
            name="csv_logs",
        )
        enable_progress_bar = False

    pl_precision = "16-mixed" if PRECISION == 16 else "32-true"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        precision=pl_precision,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
        ],
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        max_steps=TRAINING_STEP,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=10,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=GRAD_CLIP,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )

    ckpt_path = (
        None
        if RESUME_CHECKPOINT in [None, "", "None", "none"]
        else RESUME_CHECKPOINT
    )
    trainer.fit(model, ckpt_path=ckpt_path)

    # Test the checkpoint selected by minimum validation loss.
    # Lightning returns metrics as ratios (0 to 1); export percent values too.
    try:
        test_results = trainer.test(model, ckpt_path="best")
        if test_results:
            metrics = test_results[0]
            if "test_UA" in metrics and "test_WA" in metrics:
                report = {
                    "session_id": SESSION_ID,
                    "seed": SEED,
                    "num_classes": model.num_classes,
                    "UA_percent": 100.0 * float(metrics["test_UA"]),
                    "WA_percent": 100.0 * float(metrics["test_WA"]),
                    "test_loss": float(metrics["test_loss"]) if "test_loss" in metrics else None,
                    "note": "Single held-out fold; not cross-validation mean",
                }
                result_path = os.path.join(saving_path, "test_metrics.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(
                    f"[Stage4] Test UA = {report['UA_percent']:.2f}%, "
                    f"WA = {report['WA_percent']:.2f}%"
                )
                print(f"[Stage4] Saved test metrics: {result_path}")
            else:
                print("[Stage4][WARN] No test_UA/test_WA in returned metrics.")
    except Exception as exc:
        print(
            f"[Stage4][WARN] trainer.test failed/skipped: {repr(exc)}"
        )

    print(f"[Stage4] DONE. ckpts/logs under: {saving_path}")


if __name__ == "__main__":
    main()
