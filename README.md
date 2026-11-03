# Speech Emotion Recognition Using Cluster-Guided Whisper Fine-Tuning and Linear Recurrent Units

This repository contains the implementation of a speech emotion recognition (SER) framework combining **Whisper**, **Cluster-Guided Fine-Tuning (CGFT)**, and **Linear Recurrent Units (LRUs)**. The method first performs supervised fine-tuning of Whisper, then generates clustering-based pseudo-labels for further encoder fine-tuning, and finally jointly trains the adapted Whisper encoder and an LRU-based classifier.

## **Dataset**

Experiments are reported on two benchmark SER datasets: **IEMOCAP** and **SAVEE**.

| Dataset | Emotion Classes | Utterances | Reported Evaluation Protocol |
| :--- | :---: | ---: | :--- |
| IEMOCAP | 4 | 5,531 | Five-fold leave-one-session-out (LOSO) cross-validation |
| SAVEE | 7 | 480 | Ten-fold cross-validation |

For **IEMOCAP**, the four classes are **angry, happy, sad, and neutral**, with *happy* and *excited* merged into the happy class. Under the reported LOSO protocol, each of the five recording sessions is used once as the held-out test session.

For **SAVEE**, all seven original emotion classes are used: **anger, disgust, fear, happiness, neutral, sadness, and surprise**. The reported evaluation uses **ten-fold cross-validation**. 

Audio recordings must be **prepared at 16 kHz before running the scripts**. Each stage then normalizes the waveform length to **30 seconds**: shorter utterances are zero-padded at the end, and utterances longer than 30 seconds are center-cropped. A validity mask is used to distinguish speech tokens from padding where applicable. The supplied scripts do **not** include a complete audio resampling or dataset-splitting utility.

The supervised stages use a `Train` split supplied by an external label JSON file, and subdivide that split approximately **80%/20%** into optimization and validation subsets. The final supervised stage also reads an external `Test` split. The outer folds (IEMOCAP sessions or SAVEE ten-fold partitions) must be constructed separately, with no overlap between the `Train` and `Test` partitions for a given fold.

Raw datasets and audio recordings are **not distributed** with this repository. Obtain the datasets through their authorized sources, prepare the audio files, and supply the corresponding fold-specific label files.

## **Models**

The framework is implemented in four sequential Python scripts:

| Stage | Script | Implementation |
| :--- | :--- | :--- |
| Stage 1 | `whisper_stage1.py` | Supervised Whisper fine-tuning using emotion labels |
| Stage 2 | `whisper_stage2.py` | FAISS K-means clustering and pseudo-label generation |
| Stage 3 | `whisper_stage3.py` | Cluster-guided Whisper fine-tuning via masked cluster prediction |
| Stage 4 | `whisper_stage4.py` | Joint Whisper–LRU fine-tuning and emotion classification |


The upstream model is **`openai/whisper-small`**. Speech representations are extracted from **encoder Layer 11**. For the IEMOCAP experiments, **encoder Layers 8–11** are fine-tuned, while the other encoder layers remain frozen. The extracted Layer-11 representations do not pass through the encoder's final LayerNorm.


**Stage 1** uses **masked temporal average pooling** and a two-layer **MLP** on top of Whisper Layer 11. It is trained with utterance-level, class-weighted cross-entropy; there is no frame-level auxiliary classification branch.

**Stage 2** loads the Whisper weights from the Stage 1 checkpoint and extracts Layer-11 representations from the designated `Train` partition. It samples **30% of valid encoder tokens per utterance** for fitting FAISS **CPU** K-means. The uploaded implementation uses three cluster sizes, **`K = [512, 1024, 2048]`**, with **200 K-means iterations**. The Stage 1 representations are computed during Stage 2; a separately saved feature file is not required.

**Stage 3** starts from the Stage 1 Whisper weights and predicts the Stage 2 cluster pseudo-labels using **three cluster-prediction heads**. A learnable mask token is injected **after encoder Layer 7**. The code sets the masking target to **6% of valid tokens**, using non-overlapping spans with lengths randomly sampled from **5 to 14 tokens** (`MASK_SPAN = 10` is the nominal setting). The loss is computed at masked, valid positions and combines the three heads using normalized weights proportional to `1 / log(K)`. **No ground-truth emotion labels are directly used in this stage.**

**Stage 4** initializes Whisper from the Stage 3 checkpoint. Its downstream classifier is:

`Whisper Layer 11 -> two-layer LRU -> masked temporal average pooling -> MLP -> emotion prediction`

The two LRU blocks use a state dimension of **128**. The MLP contains LayerNorm, a hidden linear layer of dimension **128**, ReLU, dropout, and an output linear layer. Whisper and the downstream model are jointly optimized using utterance-level, class-weighted cross-entropy, with **no auxiliary frame-level loss**.

## **Operations Guide**

Run the four scripts **in order for each evaluation fold**. They are not an automated multi-fold pipeline. The Stage 1–4 scripts use **project-relative paths** for datasets, labels, checkpoints, and output directories. Before running each fold, ensure that the dataset paths and checkpoint filenames are correctly configured. Stage 2 and Stage 3 must use the same selected Stage 1 checkpoint, while Stage 4 must load the corresponding Stage 3 checkpoint. All scripts should be executed from the repository root directory.

### **1. Prepare the audio and fold labels**

Place the preprocessed **16-kHz** audio files in an accessible directory and prepare a JSON file with `Train` and `Test` mappings from audio filenames to emotion labels. For example, the scripts accept the following *illustrative* structure (filenames and labels are placeholders):

```json
{
  "Train": {"train_example.wav": "neutral"},
  "Test": {"test_example.wav": "sad"}
}
```


### **2. Stage 1 — Supervised Whisper Fine-Tuning**

```bash
python whisper_stage1.py \
  --datadir /path/to/Audio_16k \
  --labelpath /path/to/fold_labels.json \
  --saving_path /path/to/stage1_checkpoints
```

Stage 1 accepts these paths as command-line arguments and trains on the label file's `Train` partition, internally reserving approximately 20% for validation (`--valid_split 0.8`). It saves the best Lightning `.ckpt` checkpoint according to **`valid_loss`**. The default configuration uses batch size **16**, learning rates **1e-5** for Whisper and **1e-4** for the classification head, and up to **15,000 optimizer steps**.

### **3. Stage 2 — FAISS K-means Clustering**

Before running, edit the constants near the beginning of `whisper_stage2.py`:

- `DATADIR`: path to 16-kHz audio files.
- `LABELPATH`: path to the **same fold's** label JSON.
- `INIT_CKPT`: Stage 1 checkpoint selected for that fold.
- `OUTPUTDIR`: output directory for pseudo-label files.
- `KS`: cluster sizes (the supplied implementation uses `[512, 1024, 2048]`).

Then run:

```bash
python whisper_stage2.py
```

The script freezes the Stage 1-initialized Whisper model, fits CPU FAISS K-means using the valid-token samples from the **external `Train` partition only**, and assigns cluster IDs to all valid tokens in that partition. Padded positions are assigned the ignore index **`-100`**. It writes:

```text
512-clus.json
1024-clus.json
2048-clus.json
all-clus.json
```

Each entry in `all-clus.json` contains the three per-token cluster-label sequences for one audio file, in the order specified by `KS`. If `KS` is changed, the per-cluster output filenames change accordingly.

### **4. Stage 3 — Cluster-Guided Fine-Tuning**

Before running, edit these constants in `whisper_stage3.py`:

- `DATADIR`: path to the audio files.
- `LABELPATH`: Stage 2's **`all-clus.json`** (not the emotion-label JSON).
- `STAGE1_CKPT_PATH`: Stage 1 checkpoint for the **same fold**.
- `SAVING_PATH`: destination for the Stage 3 checkpoints.
- `KS`: must **exactly match** Stage 2's cluster sizes and ordering.

Then run:

```bash
python whisper_stage3.py
```

Stage 3 loads Stage 2 pseudo-labels and the Stage 1 Whisper weights. It performs masked cluster prediction without directly using emotion labels. The uploaded configuration uses batch size **8**, a **6%** token masking target, trainable Whisper Layers **8–11**, and up to **20,000 training steps**. It saves a best checkpoint selected by **`val_loss`**, as well as a last checkpoint, in `SAVING_PATH`.

**Important:** The Stage 2 `INIT_CKPT` and Stage 3 `STAGE1_CKPT_PATH` defaults in the uploaded files point to different locations. Set both to the **same intended Stage 1 checkpoint** before running the two stages for a fold.

### **5. Stage 4 — Joint Whisper–LRU Fine-Tuning**

Before running, edit these constants in `whisper_stage4.py`:

- `DATADIR`: path to the 16-kHz audio files.
- `LABELPATH`: fold-specific emotion-label JSON containing both `Train` and `Test`.
- `STAGE3_CKPT_PATH`: Stage 3 checkpoint for the same fold.
- `SESSION_ID`: identifier used in the output directory name (rename or reinterpret this for SAVEE folds).
- `BASE_SAVING_DIR`: base directory for checkpoints and metrics.

Then run:

```bash
python whisper_stage4.py
```

Stage 4 trains Whisper Layers 8–11 jointly with the **two-layer LRU, masked mean pooling, and MLP**. The uploaded configuration uses batch size **16**, LRU state dimension **128**, learning rates **5e-6** (encoder) and **3e-4** (downstream), gradient accumulation of **2**, and up to **10,000 training steps**. It selects the best checkpoint using **`valid_loss`** and then attempts to evaluate that checkpoint on the external `Test` split.

Stage 4 reports **weighted accuracy (`test_WA`)** and **unweighted accuracy (`test_UA`)**. The best checkpoint is selected based on the minimum validation loss and evaluated on the held-out test set. The evaluation results are saved in `test_metrics.json`. The scripts do not provide a complete automatic cross-fold aggregation utility.


### **6. Dataset-specific configuration**

The provided scripts are configured primarily for **IEMOCAP Session 1**, not as separate ready-to-run IEMOCAP and SAVEE implementations. To run all folds, repeat the four stages with correctly matched data partitions, checkpoint paths and output folders. For **SAVEE seven-class ten-fold cross-validation**, prepare ten external fold label files and update each stage for each fold. If the SAVEE experiment uses a different cluster-size configuration, change `KS` consistently in **both** Stage 2 and Stage 3, and regenerate the pseudo-labels. The SAVEE-specific cluster settings and split-generation code are **not established by these four uploaded scripts**.

Avoid placing evaluation-test examples in the training partition or using them when fitting K-means. No raw dataset files, experiment-specific checkpoints or trained weights are supplied here.

## **Environment Specifications**

### Hardware Configuration

- **GPU used in the reported experiments:** NVIDIA A100 80 GB PCIe.

### Software Environment

The previously recorded experimental environment is summarized below. Exact dependency versions should be checked against the working environment before a reproducibility release; the Python scripts themselves do not pin library versions.

| Component | Previously Recorded Version / Requirement |
| :--- | :--- |
| Python | 3.10.19 |
| PyTorch | 2.0.1+cu118 |
| torchaudio | 2.0.2 |
| CUDA | 11.8 |
| PyTorch Lightning | 2.0.2 |
| transformers | Required; verify installed version |
| faiss-cpu | Required for Stage 2; verify installed version |
| numpy | Required; verify installed version |
| soundfile | Required; verify installed version |
| tqdm | Required for Stage 2; verify installed version |

The implementation uses `WhisperModel` and `WhisperFeatureExtractor` from Hugging Face Transformers. Some encoder calls include compatibility handling for different library versions, but equivalent results are not guaranteed across untested environments.

## **Result**

The following are the **reported experimental results**, rather than values recomputed from the four source files. **UA** represents unweighted accuracy (unweighted average recall), and **WA** represents weighted accuracy.

| Dataset | Evaluation Protocol | UA (%) | WA (%) |
| :--- | :--- | ---: | ---: |
| IEMOCAP | Four-class, five-fold session-wise LOSO | **77.45** | **78.03** |
| SAVEE | Seven-class, ten-fold cross-validation | **87.13** | **88.02** |

These aggregate results require the corresponding fold-specific experiment outputs for independent verification. The provided Stage 4 script evaluates a configured fold and prints/logs its metrics; it does not itself produce the above cross-validation summary table. The fold-level predictions, model weights, and full dataset-preparation code are not included in the four uploaded scripts.
