# generate_labels_sessionwise.py
import os
import json
from pathlib import Path


def in_session(speakerset, audioname):
    """
    speakerset: ['M1', 'F1'] 等
    audioname: 例如 'Ses01F_impro01_F000.wav'
    - audio_gender = 'F'
    - audio_session = '1'
    """
    audio_gender = audioname[-8]  # 倒数第 8 个字符：M/F
    audio_session = audioname[4]  # 第 5 个字符：Session号 (1..5)
    for speaker in speakerset:
        gender = speaker[0]
        session = speaker[1]
        if gender == audio_gender and session == audio_session:
            return True
    return False


def main():
    metalabel_path = Path('metalabel.json')
    if not metalabel_path.exists():
        raise FileNotFoundError("metalabel.json not found. Run gen_meta_label.py first.")

    with open(metalabel_path, 'r', encoding='utf-8') as f:
        metalabel = json.load(f)

    audio16k_dir = Path('Audio_16k')
    if not audio16k_dir.exists():
        raise FileNotFoundError("Audio_16k not found. Run make_16k.py first.")

    out_dir = Path('labels_sess')
    out_dir.mkdir(exist_ok=True)

    # 简化后的情感标签映射
    labeldict = {
        'ang': 'anger',
        'hap': 'happy',
        'exc': 'happy',
        'sad': 'sad',
        'neu': 'neutral'
        # 其他情感被丢弃（与原版一致）
    }

    print("Generating session-wise Train/Test labels into labels_sess/label_*.json")

    for i in range(5):
        sess = i + 1
        # 该 Session 的说话人作为 Test
        test_speakers = [f'M{sess}', f'F{sess}']

        labels = {
            'Train': {},
            'Val': {},   # 原代码中没有填充，保持为空结构即可
            'Test': {}
        }

        for audio in os.listdir(audio16k_dir):
            if audio not in metalabel:
                continue

            label_key = metalabel[audio]  # 如 'ang', 'hap', 'exc', ...
            if label_key not in labeldict:
                continue  # 丢弃不在 labeldict 内的情感

            label = labeldict[label_key]

            if in_session(test_speakers, audio):
                labels['Test'][audio] = label
            else:
                labels['Train'][audio] = label

        out_path = out_dir / f'label_{i+1}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=4)

        print(f"Session {i+1}: Train={len(labels['Train'])}, Test={len(labels['Test'])}, saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
