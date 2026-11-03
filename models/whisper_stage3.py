import os
import json
import math
import inspect

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import WhisperModel, WhisperFeatureExtractor


# ============================================================
# Paths: relative to the repository root
# ============================================================

# This script is located in <repository>/models/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATADIR = os.path.join(PROJECT_ROOT, "dataset", "Audio_16k")

# Stage 2 generated pseudo-labels for Session 1.
LABELPATH = os.path.join(PROJECT_ROOT, "outputs", "stage2_sess1", "all-clus.json")

# Same Stage 1 checkpoint as Stage 2 INIT_CKPT.
# Replace the filename with the actual selected Stage 1 checkpoint if different.
STAGE1_CKPT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "stage1_sess1",
    "whisper-s1-layer11-out11-epoch=05-valid_loss=1.2547-valid_acc=0.7385.ckpt",
)

# Stage 3 checkpoint output directory.
SAVING_PATH = os.path.join(PROJECT_ROOT, "outputs", "stage3_sess1")


# ============================================================
# 最终固定实验配置
# ============================================================

WHISPER_MODEL_NAME = "openai/whisper-small"

SAMPLE_RATE = 16000
TARGET_SAMPLES = 480000
T_ENC = 1500
IGNORE_INDEX = -100

TRAIN_LAYER_START = 8
TRAIN_LAYER_END = 11
OUTPUT_LAYER = 11
MASK_INJECT_AFTER_LAYER = 7

KS = [512, 1024, 2048]

BATCH_SIZE = 8
NUM_WORKERS = 8
TRAINING_STEP = 20000
PRECISION = 16

LR_ENC = 1e-5
LR_HEAD = 7e-5
WEIGHT_DECAY = 0.02
VALID_SPLIT = 0.8
SEED = 42

MASK_PROB = 0.06
MASK_SPAN_MIN = 5
MASK_SPAN_MAX = 14  # 每段随机取 5–14 个 token（包含两端）。

HEAD_DROPOUT = 0.2
LABEL_SMOOTHING = 0.05

EARLY_PATIENCE = 10
EARLY_MIN_DELTA = 1e-4
GRAD_CLIP = 1.0

WARMUP_RATIO = 0.05
FLAT_RATIO = 0.05


# ============================================================
# Stage2 pseudo-label loading
# ============================================================

def load_all_clus_json(cluster_json_path: str):
    """读取最终 Stage2 的 all-clus.json，并固定验证 K=[512,1024,2048]。"""
    if not os.path.isfile(cluster_json_path):
        raise FileNotFoundError(
            f"[Stage3] 找不到 Stage2 pseudo-label 文件：{cluster_json_path}"
        )

    with open(cluster_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            "[Stage3] all-clus.json 应为 dict[name] -> List[List[int]]。"
        )

    cluster_label_dict = {}

    for name, seq_lists in raw.items():
        if str(name).startswith("__"):
            continue

        if not isinstance(seq_lists, list) or len(seq_lists) != len(KS):
            raise ValueError(
                f"[Stage3] {name} 的 head 数不正确："
                f"got={len(seq_lists) if isinstance(seq_lists, list) else type(seq_lists)}, "
                f"expected={len(KS)}"
            )

        tensors = []
        reference_length = None

        for head_idx, (seq, k) in enumerate(zip(seq_lists, KS)):
            if not isinstance(seq, list):
                raise ValueError(
                    f"[Stage3] {name}, head={head_idx} 的标签序列不是 list。"
                )

            if reference_length is None:
                reference_length = len(seq)
            elif len(seq) != reference_length:
                raise ValueError(
                    f"[Stage3] {name} 的不同 cluster heads 标签长度不一致。"
                )

            arr = torch.tensor(seq, dtype=torch.long)
            valid = arr[arr != IGNORE_INDEX]

            if valid.numel() > 0:
                min_id = int(valid.min().item())
                max_id = int(valid.max().item())
                if min_id < 0:
                    raise ValueError(
                        f"[Stage3] {name}, head={head_idx} 存在非法负标签 {min_id}。"
                    )
                if max_id >= k:
                    raise ValueError(
                        f"[Stage3] {name}, head={head_idx}: label max={max_id} >= K={k}"
                    )

            tensors.append(arr)

        cluster_label_dict[name] = torch.stack(tensors, dim=0)

    if not cluster_label_dict:
        raise ValueError("[Stage3] all-clus.json 中没有有效样本。")

    first_name = next(iter(cluster_label_dict))
    print(f"[Stage3] Loaded pseudo labels: {len(cluster_label_dict)} utterances")
    print(f"[Stage3] K = {KS}")
    print(
        f"[Stage3] Example label shape: "
        f"{tuple(cluster_label_dict[first_name].shape)}"
    )

    return cluster_label_dict


# ============================================================
# Audio preprocessing
# ============================================================

def zero_pad_or_center_crop_to_30s(wav: np.ndarray, valid_len: int):
    """
    与最终 Stage1 / Stage2 保持一致：
        <30s  -> 末尾 zero-padding
        =30s  -> 不变
        >30s  -> center crop
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    valid_len = int(max(0, min(valid_len, wav.shape[0])))
    base = wav[:valid_len] if valid_len > 0 else wav
    length = int(base.shape[0])

    if length <= 0:
        return np.zeros((TARGET_SAMPLES,), dtype=np.float32), 0

    if length >= TARGET_SAMPLES:
        start = (length - TARGET_SAMPLES) // 2
        segment = base[start:start + TARGET_SAMPLES]
        return segment.astype(np.float32, copy=False), TARGET_SAMPLES

    output = np.zeros((TARGET_SAMPLES,), dtype=np.float32)
    output[:length] = base
    return output, length


# ============================================================
# Mask generation
# ============================================================

def make_time_mask_tok_valid(valid_mask_tok: torch.Tensor) -> torch.Tensor:
    """
    在有效 token 区域生成 span masking。

    最终配置：MASK_PROB=0.06，每段随机取 5–14 个 token。
    不同 span 不重叠。
    """
    if valid_mask_tok.dim() != 2:
        raise ValueError("[Stage3] valid_mask_tok 必须是 [B,T]。")

    batch_size, token_length = valid_mask_tok.shape
    device = valid_mask_tok.device
    mask = torch.zeros(
        batch_size, token_length, dtype=torch.bool, device=device
    )

    for b in range(batch_size):
        valid_len = int(valid_mask_tok[b].long().sum().item())
        valid_len = max(1, min(valid_len, token_length))

        if valid_len <= 1:
            continue

        num_mask = int(valid_len * MASK_PROB)
        if num_mask <= 0:
            continue

        masked = 0
        tries = 0

        while masked < num_mask and tries < 10 * num_mask:
            span = int(
                torch.randint(
                    MASK_SPAN_MIN, MASK_SPAN_MAX + 1,
                    (1,), device=device
                ).item()
            )
            span = min(span, valid_len)

            start_max = max(1, valid_len - span + 1)
            start = int(
                torch.randint(
                    0, start_max, (1,), device=device
                ).item()
            )
            end = min(valid_len, start + span)

            if mask[b, start:end].any():
                tries += 1
                continue

            mask[b, start:end] = True
            masked += end - start

    return mask & valid_mask_tok


# ============================================================
# Dataset
# ============================================================

class Stage3Dataset(Dataset):
    """Stage3 样本由 Stage2 all-clus.json 中的文件名决定。"""

    def __init__(self, datadir: str, cluster_label_dict):
        super().__init__()
        self.datadir = datadir.rstrip("/")
        self.cluster_label_dict = cluster_label_dict
        self.names = sorted(cluster_label_dict.keys())
        print(f"[Stage3] Total {len(self.names)} examples.")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        wav_path = os.path.join(self.datadir, name)

        if not os.path.isfile(wav_path):
            raise FileNotFoundError(
                f"[Stage3] 找不到音频文件：{wav_path} (name={name})"
            )

        wav, sr = sf.read(wav_path)

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"[Stage3] {wav_path} 采样率={sr}，预期={SAMPLE_RATE}"
            )

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        wav = wav.astype(np.float32)
        wav_tensor = torch.from_numpy(wav)
        wav_len = int(wav_tensor.numel())
        cluster_labels = self.cluster_label_dict[name]

        return wav_tensor, wav_len, cluster_labels, name


# ============================================================
# Stage3 PTAPT model
# ============================================================

class WhisperPTAPTModule(pl.LightningModule):
    """
    最终 Stage3：
      1. Whisper-small
      2. 加载 Stage1 Whisper 权重
      3. 冻结全部 Whisper
      4. 解冻 encoder Layers 8-11 + encoder final LayerNorm
      5. Layer 7 后注入 learnable mask token
      6. 输出 Layer 11 representation
      7. 三个 heads 预测 K=512/1024/2048 pseudo-labels
      8. 仅在 masked & valid tokens 上计算 weighted CE
    """

    def __init__(self):
        super().__init__()

        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.maxstep = TRAINING_STEP

        # Stage2 pseudo-labels
        cluster_label_dict = load_all_clus_json(LABELPATH)

        # Whisper-small
        self.whisper = WhisperModel.from_pretrained(WHISPER_MODEL_NAME)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            WHISPER_MODEL_NAME
        )
        d_model = int(self.whisper.config.d_model)

        # Load Stage1 adapted Whisper weights
        self._load_stage1_checkpoint(STAGE1_CKPT_PATH)

        # Freeze all, then unfreeze encoder Layers 8-11.
        for p in self.whisper.parameters():
            p.requires_grad = False

        for layer in self.whisper.encoder.layers[
            TRAIN_LAYER_START - 1:TRAIN_LAYER_END
        ]:
            for p in layer.parameters():
                p.requires_grad = True

        # 保留原 Stage1 wrapper 的行为。
        for p in self.whisper.encoder.layer_norm.parameters():
            p.requires_grad = True

        # Learnable mask embedding
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # 80/20 train-validation split
        full_data = Stage3Dataset(DATADIR, cluster_label_dict)
        n_total = len(full_data)
        n_train = int(n_total * VALID_SPLIT)
        n_val = n_total - n_train

        self.train_data, self.val_data = random_split(
            full_data,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED),
        )
        print(f"[Stage3] 数据划分：train={n_train}, val={n_val}")

        # Three cluster-prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(p=HEAD_DROPOUT),
                nn.Linear(d_model, k),
            )
            for k in KS
        ])

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=LABEL_SMOOTHING,
            ignore_index=IGNORE_INDEX,
        )

        # 保留原代码：w_k ∝ 1/log(K)
        raw_weights = [1.0 / np.log(k + 1e-8) for k in KS]
        total_weight = sum(raw_weights)
        normalized_weights = [w / total_weight for w in raw_weights]
        self.register_buffer(
            "head_weights",
            torch.tensor(normalized_weights, dtype=torch.float32),
        )

        print(f"[Stage3] head Ks={KS}")
        print(f"[Stage3] head weights={normalized_weights}")

        trainable_names = [
            name for name, p in self.whisper.named_parameters()
            if p.requires_grad
        ]
        print(
            f"[Stage3] Trainable Whisper parameter tensors="
            f"{len(trainable_names)}"
        )

    # --------------------------------------------------------
    # Stage1 checkpoint
    # --------------------------------------------------------

    def _load_stage1_checkpoint(self, ckpt_path: str):
        """
        最终 Stage1 Lightning checkpoint 中 WhisperModel 参数前缀为：
            whisper.model.*
        去掉该前缀后加载到 Stage3 的 WhisperModel。
        """
        ckpt_path = os.path.expanduser(ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"[Stage3] 找不到 Stage1 ckpt：{ckpt_path}"
            )
        if not ckpt_path.endswith(".ckpt"):
            raise ValueError(
                f"[Stage3] Stage1 checkpoint 必须是 .ckpt：{ckpt_path}"
            )

        print(f"[Stage3] Loading Stage1 checkpoint:\n{ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        if not isinstance(state_dict, dict):
            raise RuntimeError("[Stage3] Invalid Stage1 checkpoint state_dict.")

        prefix = "whisper.model."
        whisper_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

        if not whisper_state:
            sample_keys = list(state_dict.keys())[:10]
            raise RuntimeError(
                "[Stage3] checkpoint 中没有找到 'whisper.model.*' 参数。\n"
                f"sample keys={sample_keys}"
            )

        missing, unexpected = self.whisper.load_state_dict(
            whisper_state,
            strict=False,
        )

        print(
            "[Stage3] Stage1 Whisper weights loaded: "
            f"source_keys={len(whisper_state)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[Stage3] missing keys sample: {missing[:10]}")
        if unexpected:
            print(f"[Stage3] unexpected keys sample: {unexpected[:10]}")

    # --------------------------------------------------------
    # Whisper manual encoder forward
    # --------------------------------------------------------

    @staticmethod
    def _get_positional_embedding(encoder, input_shape, device, dtype):
        """保留轻量兼容逻辑，适配不同 transformers 版本。"""
        if not hasattr(encoder, "embed_positions"):
            return None

        try:
            pos = encoder.embed_positions(input_shape)
        except TypeError:
            dummy = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=device,
            )
            pos = encoder.embed_positions(dummy)

        return pos.to(device=device, dtype=dtype)

    @staticmethod
    def _build_additive_attention_mask(valid_mask_tok, dtype, device):
        """[B,T] valid mask -> [B,1,1,T] additive attention mask。"""
        if valid_mask_tok is None:
            return None

        if valid_mask_tok.dtype != torch.bool:
            valid_mask_tok = valid_mask_tok.bool()

        valid_mask_tok = valid_mask_tok.to(device=device)
        invalid = ~valid_mask_tok
        attention_mask = invalid.to(dtype=dtype) * (-1e4)
        return attention_mask[:, None, None, :]

    @staticmethod
    def _call_encoder_layer(layer, hidden_states, attention_mask_4d=None):
        """兼容不同 transformers 版本中的 WhisperEncoderLayer 参数。"""
        signature = inspect.signature(layer.forward)
        kwargs = {}

        if "attention_mask" in signature.parameters:
            kwargs["attention_mask"] = attention_mask_4d
        if "layer_head_mask" in signature.parameters:
            kwargs["layer_head_mask"] = None
        if "output_attentions" in signature.parameters:
            kwargs["output_attentions"] = False

        output = layer(hidden_states, **kwargs)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _encode_features_with_mask(
        self,
        input_features: torch.Tensor,
        mask_tok: torch.Tensor,
        valid_mask_tok: torch.Tensor,
    ):
        """
        手动执行 Whisper encoder 到 Layer 11：

        conv frontend
          -> Layers 1-7
          -> mask-token injection
          -> Layers 8-11
          -> Layer 11 representation
        """
        encoder = self.whisper.encoder

        hidden_states = F.gelu(encoder.conv1(input_features))
        hidden_states = F.gelu(encoder.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)  # [B,T,C]

        pos = self._get_positional_embedding(
            encoder,
            hidden_states.size()[:-1],
            hidden_states.device,
            hidden_states.dtype,
        )
        if pos is not None:
            hidden_states = hidden_states + pos

        dropout_p = float(getattr(encoder, "dropout", 0.0))
        if dropout_p > 0:
            hidden_states = F.dropout(
                hidden_states,
                p=dropout_p,
                training=self.training,
            )

        batch_size, token_length, _ = hidden_states.shape

        if token_length != T_ENC:
            raise RuntimeError(
                f"[Stage3] Whisper token length={token_length}, expected={T_ENC}"
            )
        if valid_mask_tok.shape != (batch_size, token_length):
            raise RuntimeError(
                f"[Stage3] valid_mask shape={tuple(valid_mask_tok.shape)}, "
                f"expected=({batch_size},{token_length})"
            )
        if mask_tok.shape != (batch_size, token_length):
            raise RuntimeError(
                f"[Stage3] mask_tok shape={tuple(mask_tok.shape)}, "
                f"expected=({batch_size},{token_length})"
            )

        attention_mask_4d = self._build_additive_attention_mask(
            valid_mask_tok,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 只执行到 Layer 11。
        for layer_idx in range(OUTPUT_LAYER):
            layer = encoder.layers[layer_idx]
            hidden_states = self._call_encoder_layer(
                layer,
                hidden_states,
                attention_mask_4d,
            )

            # 完成 Layer 7 后，注入 learnable mask embedding。
            # 使用 torch.where 避免 PyTorch 2.0.1 + CUDA 下
            # Boolean advanced-indexing assignment 的广播问题。
            if layer_idx + 1 == MASK_INJECT_AFTER_LAYER:
                mask_tok = mask_tok.bool()

                mask_value = self.mask_token.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ).view(1, 1, -1)

                hidden_states = torch.where(
                    mask_tok.unsqueeze(-1),
                    mask_value,
                    hidden_states,
                )

        return hidden_states

    def encode_with_hidden_mask(
        self,
        wav_batch: torch.Tensor,
        mask_tok: torch.Tensor,
        valid_mask_tok: torch.Tensor,
    ):
        """waveform -> Whisper log-Mel -> masked Layer-11 representation。"""
        wav_cpu = wav_batch.detach().cpu().numpy()

        input_features = self.feature_extractor(
            wav_cpu,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=TARGET_SAMPLES,
        ).input_features.to(self.device)

        return self._encode_features_with_mask(
            input_features,
            mask_tok,
            valid_mask_tok,
        )

    # --------------------------------------------------------
    # Collate
    # --------------------------------------------------------

    def _collate(self, batch):
        wavs, wav_lens, labels, names = zip(*batch)

        # Audio -> exactly 30 s
        wav_30_list = []
        for wav, wav_len in zip(wavs, wav_lens):
            wav_np = wav.detach().cpu().numpy().astype(np.float32)
            wav_30s, _ = zero_pad_or_center_crop_to_30s(
                wav_np,
                int(wav_len),
            )
            wav_30_list.append(torch.from_numpy(wav_30s))

        wav_batch = torch.stack(wav_30_list, dim=0).float()

        # Pseudo labels + valid mask
        batch_size = len(labels)
        label_batch = torch.full(
            (batch_size, len(KS), T_ENC),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )
        valid_mask = torch.zeros(
            (batch_size, T_ENC),
            dtype=torch.bool,
        )

        for i, label in enumerate(labels):
            if label.dim() != 2:
                raise RuntimeError(
                    f"[Stage3] label dim={label.dim()}, expected=2"
                )
            if label.size(0) != len(KS):
                raise RuntimeError(
                    f"[Stage3] label heads={label.size(0)}, expected={len(KS)}"
                )

            label_length = int(label.shape[1])
            copy_length = min(label_length, T_ENC)
            label_batch[i, :, :copy_length] = label[:, :copy_length]

            # 三个 head 在该位置都不是 -100 时，才视为有效 token。
            valid_mask[i, :copy_length] = (
                label_batch[i, :, :copy_length] != IGNORE_INDEX
            ).all(dim=0)

        return wav_batch, label_batch, valid_mask, list(names)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    # --------------------------------------------------------
    # Optimizer + scheduler
    # --------------------------------------------------------

    def configure_optimizers(self):
        encoder_params = [
            p for p in self.whisper.parameters()
            if p.requires_grad
        ]

        optimizer = optim.AdamW(
            [
                {"params": encoder_params, "lr": LR_ENC},
                {"params": [self.mask_token], "lr": LR_ENC},
                {"params": self.heads.parameters(), "lr": LR_HEAD},
            ],
            weight_decay=WEIGHT_DECAY,
        )

        num_training_steps = self.maxstep
        num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
        num_flat_steps = int(FLAT_RATIO * num_training_steps)

        def lambda_lr(step: int):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))

            if step < num_warmup_steps + num_flat_steps:
                return 1.0

            return max(
                0.0,
                float(num_training_steps - step)
                / float(
                    max(
                        1,
                        num_training_steps
                        - (num_warmup_steps + num_flat_steps),
                    )
                ),
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
    # Masked cluster prediction loss
    # --------------------------------------------------------

    def _compute_loss(self, reps, mask_tok, label_batch, valid_mask):
        """只在 masked & valid token 上计算三头 weighted CE。"""
        batch_size, token_length, dim = reps.shape
        reps_flat = reps.reshape(batch_size * token_length, dim)

        use_flat = (mask_tok & valid_mask).reshape(
            batch_size * token_length
        )

        if use_flat.sum() == 0:
            return reps_flat.sum() * 0.0

        masked_reps = reps_flat[use_flat]
        total_loss = 0.0

        for head_idx, head in enumerate(self.heads):
            labels_flat = label_batch[:, head_idx, :].reshape(
                batch_size * token_length
            )
            masked_labels = labels_flat[use_flat]
            logits = head(masked_reps)
            loss_i = self.criterion(logits, masked_labels)
            total_loss = total_loss + self.head_weights[head_idx] * loss_i

        return total_loss

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------

    def training_step(self, batch, batch_idx):
        wav_batch, label_batch, valid_mask, _names = batch
        wav_batch = wav_batch.to(self.device)
        label_batch = label_batch.to(self.device)
        valid_mask = valid_mask.to(self.device)

        mask_tok = make_time_mask_tok_valid(valid_mask)

        # AMP 由 Lightning Trainer(precision="16-mixed") 统一管理。
        reps = self.encode_with_hidden_mask(
            wav_batch,
            mask_tok,
            valid_mask,
        )
        loss = self._compute_loss(
            reps,
            mask_tok,
            label_batch,
            valid_mask,
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        wav_batch, label_batch, valid_mask, _names = batch
        wav_batch = wav_batch.to(self.device)
        label_batch = label_batch.to(self.device)
        valid_mask = valid_mask.to(self.device)

        mask_tok = make_time_mask_tok_valid(valid_mask)
        reps = self.encode_with_hidden_mask(
            wav_batch,
            mask_tok,
            valid_mask,
        )
        loss = self._compute_loss(
            reps,
            mask_tok,
            label_batch,
            valid_mask,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(SAVING_PATH, exist_ok=True)

    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    precision = "16-mixed" if PRECISION == 16 else "32-true"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    print("[Stage3] ========================================")
    print(f"[Stage3] Whisper = {WHISPER_MODEL_NAME}")
    print(f"[Stage3] Train Layers = {TRAIN_LAYER_START}-{TRAIN_LAYER_END}")
    print(f"[Stage3] Output Layer = {OUTPUT_LAYER}")
    print(
        f"[Stage3] Mask injection after Layer "
        f"{MASK_INJECT_AFTER_LAYER}"
    )
    print(f"[Stage3] Mask prob = {MASK_PROB}")
    print(
        f"[Stage3] Random mask span = "
        f"{MASK_SPAN_MIN}–{MASK_SPAN_MAX} tokens"
    )
    print(f"[Stage3] K = {KS}")
    print(
        f"[Stage3] Train/Val = {VALID_SPLIT:.1f}/"
        f"{1.0 - VALID_SPLIT:.1f}"
    )
    print("[Stage3] Audio = <30s zero-pad, >30s center-crop")
    print("[Stage3] ========================================")

    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVING_PATH,
        filename=(
            "whisper-s3-layer11-"
            "{epoch:02d}-{val_loss:.4f}"
        ),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EARLY_PATIENCE,
        min_delta=EARLY_MIN_DELTA,
        verbose=True,
    )

    model = WhisperPTAPTModule()

    trainer = Trainer(
        precision=precision,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
        ],
        max_steps=TRAINING_STEP,
        accelerator=accelerator,
        devices=1,
        logger=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        gradient_clip_val=GRAD_CLIP,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )

    trainer.fit(model)

    print(
        f"[Stage3] 训练完成，best/last checkpoint 保存在：{SAVING_PATH}"
    )


if __name__ == "__main__":
    main()
