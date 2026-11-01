# whisper_stage2.py


import os
import json
import math
from pathlib import Path
from typing import Optional, Tuple
import faiss
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from transformers import WhisperModel, WhisperFeatureExtractor
from tqdm import tqdm


# ============================================================
# Paths (portable, repository-relative)
# ============================================================

# This script is located at <repository>/models/whisper_stage2.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default fold: IEMOCAP Session 1. Override without editing the script:
#     IEMOCAP_SESSION_ID=2 python models/whisper_stage2.py
SESSION_ID = int(os.environ.get("IEMOCAP_SESSION_ID", "1"))
if SESSION_ID not in range(1, 6):
    raise ValueError("IEMOCAP_SESSION_ID must be an integer from 1 to 5.")

DATADIR = str(PROJECT_ROOT / "dataset" / "Audio_16k")
LABELPATH = str(
    PROJECT_ROOT / "dataset" / "labels_sess" / f"label_{SESSION_ID}.json"
)
OUTPUTDIR = str(PROJECT_ROOT / "outputs" / f"stage2_sess{SESSION_ID}")

# Optional explicit checkpoint selection, for example:
#     STAGE1_CKPT=/path/to/your/stage1-best.ckpt python models/whisper_stage2.py
# If unset, exactly one .ckpt must exist in outputs/stage1_sess{SESSION_ID}/.
INIT_CKPT = os.environ.get("STAGE1_CKPT", "").strip()


def resolve_stage1_ckpt() -> str:
    """Select the Stage 1 checkpoint without hard-coding personal paths.

    When multiple checkpoints exist, require an explicit selection rather
    than guessing which one is the validation-best checkpoint.
    """
    if INIT_CKPT:
        checkpoint = Path(INIT_CKPT).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"STAGE1_CKPT does not exist: {checkpoint}"
            )
        return str(checkpoint)

    ckpt_dir = PROJECT_ROOT / "outputs" / f"stage1_sess{SESSION_ID}"
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise FileNotFoundError(
            f"No Stage 1 checkpoint found in {ckpt_dir}. "
            "Train Stage 1 first or specify STAGE1_CKPT."
        )
    raise RuntimeError(
        f"Found {len(candidates)} Stage 1 checkpoints in {ckpt_dir}. "
        "Specify STAGE1_CKPT to select the correct checkpoint for this fold."
    )


# ============================================================
# 最终固定实验配置
# ============================================================

WHISPER_MODEL_NAME = "openai/whisper-small"

SAMPLE_RATE = 16000
HOP_LENGTH = 160

TARGET_SEC = 30.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SEC)
MAX_MEL_FRAMES = 3000
OUTPUT_LAYER = 11

KS = [512, 1024, 2048]

SAMPLE_RATIO = 0.30

BATCH_SIZE = 4
NUM_WORKERS = 4

KMEANS_MAX_ITER = 200
FAISS_NREDO = 1
FAISS_MAX_POINTS_PER_CENTROID = 2000
FAISS_VERBOSE = False

SEED = 42
IGNORE_INDEX = -100


# ============================================================
# Utility
# ============================================================

def compute_valid_token_length(wav_len_samples: int) -> int:
    """
    将有效 waveform 长度映射到 Whisper encoder 的有效 token 长度。

    Whisper:
        mel frames    ~= ceil(samples / hop_length)
        encoder token ~= ceil(mel_frames / 2)
    """
    if wav_len_samples <= 0:
        return 1

    n_mel_valid = int(
        math.ceil(wav_len_samples / float(HOP_LENGTH))
    )
    n_mel_valid = min(
        MAX_MEL_FRAMES,
        max(1, n_mel_valid),
    )

    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)


def build_token_mask(
    valid_wav_lens: torch.Tensor,
    token_length: int,
    device,
) -> torch.Tensor:
    """
    构造 [B, T] token mask。

    True  = 有效语音区域
    False = zero-padding 区域
    """
    if not torch.is_tensor(valid_wav_lens):
        valid_wav_lens = torch.tensor(valid_wav_lens)

    valid_wav_lens = (
        valid_wav_lens
        .detach()
        .cpu()
        .long()
        .view(-1)
    )

    valid_token_lengths = [
        min(
            compute_valid_token_length(int(wav_len.item())),
            token_length,
        )
        for wav_len in valid_wav_lens
    ]

    valid_token_lengths = torch.tensor(
        valid_token_lengths,
        device=device,
        dtype=torch.long,
    )

    time_ids = torch.arange(
        token_length,
        device=device,
    ).view(1, token_length)

    return time_ids < valid_token_lengths.view(-1, 1)


# ============================================================
# Stage1 checkpoint loading
# ============================================================

def load_stage1_ckpt_into_whisper_model(
    whisper_model: WhisperModel,
    ckpt_path: str,
):
    """
    从最终 Stage1 Lightning checkpoint 中只读取 whisper.model.*
    并加载到 Stage2 的 WhisperModel。

    Stage1 的分类头与 pooling 参数不会加载。
    """
    if ckpt_path in [None, "None", ""]:
        raise ValueError(
            "INIT_CKPT 不能为空。Stage2 必须加载 Stage1 训练后的 checkpoint。"
        )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Stage1 checkpoint 不存在：{ckpt_path}"
        )

    print(f"[Stage2] Loading Stage1 checkpoint:\n{ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    state_dict = ckpt.get(
        "state_dict",
        ckpt,
    )

    prefix = "whisper.model."

    whisper_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }

    if len(whisper_state) == 0:
        raise RuntimeError(
            "没有在 checkpoint 中找到 'whisper.model.*' 参数。"
            "请确认该 checkpoint 是由最终 Stage1 代码生成的。"
        )

    missing, unexpected = whisper_model.load_state_dict(
        whisper_state,
        strict=False,
    )

    print(
        "[Stage2] Stage1 Whisper weights loaded. "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


# ============================================================
# Whisper encoder
# ============================================================

class WhisperEncoderWrapper(nn.Module):
    """
    Stage2 固定配置：

        - Whisper-small
        - 加载 Stage1 训练后的 Whisper 权重
        - Stage2 中 Whisper 完全冻结
        - 输出 encoder 第 11 层 hidden states
    """

    def __init__(self):
        super().__init__()

        self.model = WhisperModel.from_pretrained(
            WHISPER_MODEL_NAME
        )

        self.feature_extractor = (
            WhisperFeatureExtractor.from_pretrained(
                WHISPER_MODEL_NAME
            )
        )

        self.hidden_size = self.model.config.d_model

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self,
        wav: torch.Tensor,
        valid_wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav_cpu = (
            wav
            .detach()
            .cpu()
            .numpy()
        )

        feats = self.feature_extractor(
            wav_cpu,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=TARGET_SAMPLES,
        )

        device = next(
            self.model.parameters()
        ).device

        input_features = feats.input_features.to(
            device
        )

        attention_mask = None

        if valid_wav_lens is not None:
            if len(valid_wav_lens) != input_features.size(0):
                raise RuntimeError(
                    "valid_wav_lens shape mismatch: "
                    f"expected {input_features.size(0)}, "
                    f"got {len(valid_wav_lens)}"
                )

            token_length = (
                int(input_features.size(-1)) + 1
            ) // 2

            attention_mask = build_token_mask(
                valid_wav_lens,
                token_length,
                input_features.device,
            )

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
                f"OUTPUT_LAYER={OUTPUT_LAYER} invalid; "
                f"hidden_states len={len(hidden_states)}"
            )

        return hidden_states[OUTPUT_LAYER]


# ============================================================
# Dataset
# ============================================================

class SplitAudioDataset(data.Dataset):
    """
    Stage2 使用 label JSON 中指定 split 的文件名读取音频。

    最终统一预处理：
        < 30 s : 末尾 zero-padding 到 30 s
        = 30 s : 不变
        > 30 s : center crop 到 30 s

    与最终 Stage1 保持一致。
    """

    def __init__(
        self,
        datadir: str,
        label_json_path: str,
        split: str = "Train",
    ):
        with open(
            label_json_path,
            "r",
            encoding="utf-8",
        ) as f:
            raw = json.load(f)

        if not (
            isinstance(raw, dict)
            and split in raw
            and isinstance(raw[split], dict)
        ):
            raise ValueError(
                f"label JSON 必须包含 split='{split}' 的 dict。"
                f"当前 keys="
                f"{list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
            )

        self.files = list(
            raw[split].keys()
        )
        self.datadir = datadir.rstrip("/")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        path = os.path.join(
            self.datadir,
            name,
        )

        wav, sr = sf.read(path)

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"[Stage2] Sample rate mismatch: "
                f"got {sr}, expected {SAMPLE_RATE} @ {name}"
            )

        wav = wav.astype(
            np.float32
        )
        wav_len = int(
            wav.shape[0]
        )

        return wav, wav_len, name

    @staticmethod
    def _zero_pad_or_center_crop_to_length(
        x: np.ndarray,
        target_len: int,
    ) -> Tuple[np.ndarray, int]:
        """
        < target_len:
            末尾补 0

        >= target_len:
            中心裁剪
        """
        x = np.asarray(
            x,
            dtype=np.float32,
        ).reshape(-1)

        length = int(
            x.shape[0]
        )

        if length <= 0:
            return (
                np.zeros(
                    (target_len,),
                    dtype=np.float32,
                ),
                0,
            )

        if length >= target_len:
            start = (
                length - target_len
            ) // 2

            segment = x[
                start:start + target_len
            ]

            return (
                segment.astype(
                    np.float32,
                    copy=False,
                ),
                target_len,
            )

        output = np.zeros(
            (target_len,),
            dtype=np.float32,
        )
        output[:length] = x

        return output, length

    def pad30s_collate(self, batch):
        processed = []

        for wav, wav_len, name in batch:
            wav = np.asarray(
                wav,
                dtype=np.float32,
            ).reshape(-1)

            wav_len = max(
                0,
                min(
                    int(wav_len),
                    wav.shape[0],
                ),
            )

            base = (
                wav[:wav_len]
                if wav_len > 0
                else wav
            )

            wav_30s, valid_len = (
                self._zero_pad_or_center_crop_to_length(
                    base,
                    TARGET_SAMPLES,
                )
            )

            processed.append(
                (
                    wav_30s,
                    int(valid_len),
                    name,
                )
            )

        return default_collate(
            processed
        )


# ============================================================
# DataLoader
# ============================================================

def build_train_loader() -> data.DataLoader:
    """
    用于收集 30% token 特征并拟合 KMeans。
    """
    dataset = SplitAudioDataset(
        DATADIR,
        LABELPATH,
        split="Train",
    )

    return data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=dataset.pad30s_collate,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


def build_infer_loader() -> data.DataLoader:
    """
    用于按原始顺序对 Train split 全部有效 token 生成伪标签。
    """
    dataset = SplitAudioDataset(
        DATADIR,
        LABELPATH,
        split="Train",
    )

    return data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=dataset.pad30s_collate,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


# ============================================================
# Collect 30% valid tokens for KMeans
# ============================================================

@torch.no_grad()
def collect_train_samples(
    model: WhisperEncoderWrapper,
    loader: data.DataLoader,
    device: str,
) -> np.ndarray:
    """
    对 Train split：

        1. 提取 Whisper Layer 11
        2. 每条 utterance 仅从 valid tokens 中随机抽样 30%
        3. 拼成 KMeans 训练矩阵 X_train: [N, C]
    """
    feats_all = []
    total_tokens = 0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc="收集 Train 特征 (30%)",
        dynamic_ncols=True,
    )

    for wav_30s, valid_len, _names in pbar:
        wav_30s = wav_30s.to(
            device,
            non_blocking=True,
        )
        valid_len = valid_len.to(
            device,
            non_blocking=True,
        )

        reps = model(
            wav_30s,
            valid_wav_lens=valid_len,
        )

        batch_size, T, _ = reps.shape

        valid_len_cpu = (
            valid_len
            .detach()
            .cpu()
            .long()
            .tolist()
        )

        for i in range(batch_size):
            valid_T = min(
                compute_valid_token_length(
                    int(valid_len_cpu[i])
                ),
                T,
            )

            if valid_T <= 0:
                continue

            x = (
                reps[i, :valid_T, :]
                .detach()
                .cpu()
                .numpy()
            )

            n_sample = int(
                math.ceil(
                    valid_T * SAMPLE_RATIO
                )
            )

            n_sample = max(
                1,
                min(
                    valid_T,
                    n_sample,
                ),
            )

            if n_sample >= valid_T:
                sampled = x
            else:
                indices = np.random.choice(
                    valid_T,
                    size=n_sample,
                    replace=False,
                )
                sampled = x[indices]

            feats_all.append(
                sampled
            )
            total_tokens += n_sample

        pbar.set_postfix(
            sampled_tokens=total_tokens
        )

    if len(feats_all) == 0:
        raise RuntimeError(
            "没有收集到任何 Whisper 特征，无法进行 KMeans。"
        )

    X_train = np.concatenate(
        feats_all,
        axis=0,
    ).astype(
        np.float32
    )

    return np.ascontiguousarray(
        X_train,
        dtype=np.float32,
    )


# ============================================================
# FAISS KMeans
# ============================================================

def fit_faiss_kmeans(
    X_train: np.ndarray,
    k: int,
):
    """
    固定最终配置：
        CPU FAISS
        niter = 200
        nredo = 1
        max_points_per_centroid = 2000
    """
    X_train = np.ascontiguousarray(
        X_train.astype(np.float32)
    )

    d = int(
        X_train.shape[1]
    )

    kmeans = faiss.Kmeans(
        d=d,
        k=int(k),
        niter=KMEANS_MAX_ITER,
        nredo=FAISS_NREDO,
        verbose=FAISS_VERBOSE,
        gpu=False,
    )

    try:
        kmeans.seed = int(SEED)
    except Exception:
        pass

    try:
        kmeans.max_points_per_centroid = int(
            FAISS_MAX_POINTS_PER_CENTROID
        )
    except Exception:
        pass

    kmeans.train(
        X_train
    )

    centroids = (
        kmeans.centroids
        .reshape(
            int(k),
            d,
        )
        .astype(
            np.float32
        )
    )

    centroids = np.ascontiguousarray(
        centroids,
        dtype=np.float32,
    )

    index = faiss.IndexFlatL2(
        d
    )
    index.add(
        centroids
    )

    return index


# ============================================================
# Pseudo-label inference and export
# ============================================================

@torch.no_grad()
def infer_and_export_json(
    model: WhisperEncoderWrapper,
    loader: data.DataLoader,
    index_list,
    device: str,
    outputdir: str,
):
    """
    对 Train split 每条语音的全部 valid tokens 分配聚类标签。

    固定输出长度为 T=1500：
        valid token   -> cluster ID
        padding token -> -100

    输出：
        512-clus.json
        1024-clus.json
        2048-clus.json
        all-clus.json
    """
    os.makedirs(
        outputdir,
        exist_ok=True,
    )

    output_by_k = [
        {}
        for _ in KS
    ]

    combined_dict = {}

    pbar = tqdm(
        loader,
        total=len(loader),
        desc="生成 Train 伪标签",
        dynamic_ncols=True,
    )

    for wav_30s, valid_len, names in pbar:
        wav_30s = wav_30s.to(
            device,
            non_blocking=True,
        )

        valid_len = valid_len.to(
            device,
            non_blocking=True,
        )

        reps = model(
            wav_30s,
            valid_wav_lens=valid_len,
        )

        batch_size, T, _ = reps.shape

        valid_len_cpu = (
            valid_len
            .detach()
            .cpu()
            .long()
            .tolist()
        )

        for i in range(batch_size):
            name = names[i]

            valid_T = min(
                compute_valid_token_length(
                    int(valid_len_cpu[i])
                ),
                T,
            )

            valid_T = max(
                0,
                int(valid_T),
            )

            labels_full_list = []

            if valid_T <= 0:
                labels_full_list = [
                    [IGNORE_INDEX] * T
                    for _ in KS
                ]

            else:
                x = (
                    reps[i, :valid_T, :]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                x = np.ascontiguousarray(
                    x,
                    dtype=np.float32,
                )

                for index in index_list:
                    _, nearest = index.search(
                        x,
                        1,
                    )

                    labels_valid = (
                        nearest[:, 0]
                        .astype(int)
                        .tolist()
                    )

                    labels_full = (
                        labels_valid
                        + [IGNORE_INDEX]
                        * (T - valid_T)
                    )

                    labels_full_list.append(
                        labels_full
                    )

            for j in range(
                len(KS)
            ):
                output_by_k[j][name] = (
                    labels_full_list[j]
                )

            combined_dict[name] = (
                labels_full_list
            )

    for j, k in enumerate(KS):
        out_path = os.path.join(
            outputdir,
            f"{k}-clus.json",
        )

        with open(
            out_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output_by_k[j],
                f,
                ensure_ascii=False,
                indent=4,
            )

        print(
            f"[Stage2] Saved K={k} labels -> {out_path}"
        )

    all_path = os.path.join(
        outputdir,
        "all-clus.json",
    )

    with open(
        all_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            combined_dict,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"[Stage2] Saved ALL labels -> {all_path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(
        SEED
    )
    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[Stage2] device = {device}")
    print(f"[Stage2] Whisper = {WHISPER_MODEL_NAME}")
    print(f"[Stage2] output layer = {OUTPUT_LAYER}")
    print(f"[Stage2] K = {KS}")
    print(f"[Stage2] KMeans sample ratio = {SAMPLE_RATIO}")
    print(
        "[Stage2] Audio processing: "
        "<30s zero-pad, >30s center-crop"
    )

    # --------------------------------------------------------
    # Whisper + Stage1 checkpoint
    # --------------------------------------------------------
    model = WhisperEncoderWrapper().to(
        device
    )

    load_stage1_ckpt_into_whisper_model(
        model.model,
        resolve_stage1_ckpt(),
    )

    model.eval()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    train_loader = build_train_loader()
    infer_loader = build_infer_loader()

    # --------------------------------------------------------
    # 30% valid-token sampling
    # --------------------------------------------------------
    print(
        "\n[Stage2] 开始收集 Train split 的 "
        "Whisper Layer 11 特征用于 KMeans ..."
    )

    X_train = collect_train_samples(
        model=model,
        loader=train_loader,
        device=device,
    )

    print(
        "[Stage2] 特征收集完成："
        f"N={X_train.shape[0]}, "
        f"dim={X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # KMeans
    # --------------------------------------------------------
    index_list = []

    for k in tqdm(
        KS,
        desc="训练 FAISS KMeans",
        dynamic_ncols=True,
    ):
        print(
            f"\n[Stage2] Training FAISS KMeans: "
            f"K={k}, "
            f"niter={KMEANS_MAX_ITER}, "
            "gpu=False"
        )

        index = fit_faiss_kmeans(
            X_train,
            k,
        )

        index_list.append(
            index
        )

    # --------------------------------------------------------
    # Pseudo-label assignment
    # --------------------------------------------------------
    print(
        "\n[Stage2] 开始为 Train split "
        "全部有效 tokens 生成伪标签 ..."
    )

    infer_and_export_json(
        model=model,
        loader=infer_loader,
        index_list=index_list,
        device=device,
        outputdir=OUTPUTDIR,
    )

    print(
        f"\n[Stage2] 完成。JSON 已写入：\n{OUTPUTDIR}"
    )


if __name__ == "__main__":
    main()
