# Dataset/IEMOCAP/make_16k.py
import os
import sys
from pathlib import Path

import librosa
import soundfile as sf


def main():
    if len(sys.argv) != 2:
        print("Usage: python make_16k.py /path/to/IEMOCAP")
        sys.exit(1)

    IEMOCAP_DIR = Path(sys.argv[1])
    out_dir = Path('Audio_16k')
    out_dir.mkdir(exist_ok=True)

    print("Downsampling / collecting IEMOCAP sentences to 16k in Audio_16k")

    count = 0
    for i in range(5):
        sess = i + 1
        current_dir = IEMOCAP_DIR / f"Session{sess}" / "sentences" / "wav"
        if not current_dir.exists():
            print(f"Warning: {current_dir} not found, skipping...")
            continue

        for full_audio_name in current_dir.rglob('*.wav'):
            audio_name = full_audio_name.name
            # 读入原始音频（保持原采样率）
            audio, sr = librosa.load(str(full_audio_name), sr=None)
            # 若不是 16k，可在此强制重采样到 16k
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            assert sr == 16000

            out_path = out_dir / audio_name
            sf.write(str(out_path), audio, 16000)
            count += 1

    print(f"Total {count} wavs written into {out_dir.resolve()}")


if __name__ == "__main__":
    main()
