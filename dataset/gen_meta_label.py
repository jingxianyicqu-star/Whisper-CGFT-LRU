# gen_meta_label.py
# 作用：从 IEMOCAP 的 EmoEvaluation 文本中提取 utterance -> emotion 映射，
#       保存为当前目录下的 metalabel.json

import os
import sys
import json
from pathlib import Path


print(">>> [gen_meta_label.py] 脚本已加载")  # 只要脚本被执行就一定会打印这一行


def main():
    print(">>> [gen_meta_label.py] main() 开始执行")

    # ---------- 1. 参数检查 ----------
    if len(sys.argv) != 2:
        print("用法: python gen_meta_label.py /path/to/IEMOCAP")
        print("Example: python gen_meta_label.py /path/to/IEMOCAP_full_release")
        print(f"当前 argv: {sys.argv}")
        sys.exit(1)

    iemocap_dir = Path(sys.argv[1])
    print(f">>> IEMOCAP 路径 = {iemocap_dir}")

    if not iemocap_dir.exists():
        print(f"[错误] IEMOCAP 目录不存在: {iemocap_dir}")
        sys.exit(1)

    # ---------- 2. 开始扫描 5 个 Session ----------
    metalabel = {}
    total_txt_files = 0
    total_lines = 0

    for sess in range(1, 6):
        label_dir = iemocap_dir / f"Session{sess}" / "dialog" / "EmoEvaluation"
        print(f">>> 检查 Session{sess} 的标签目录: {label_dir}")

        if not label_dir.exists():
            print(f"    [警告] 目录不存在, 跳过该 Session: {label_dir}")
            continue

        # 遍历该 Session 下所有 txt
        for labelfile in sorted(label_dir.rglob("*.txt")):
            total_txt_files += 1
            print(f"    读取标签文件: {labelfile}")
            with open(labelfile, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    # 典型格式类似：
                    # [0.93 - 3.12]\tSes01F_impro01_F000\tang\t[...]
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    utt_id = parts[1]   # 如 Ses01F_impro01_F000
                    emo = parts[2]      # 如 ang, hap, sad, neu, ...
                    if utt_id.startswith("Ses"):
                        metalabel[utt_id + ".wav"] = emo

    # ---------- 3. 保存结果 ----------
    out_path = Path("metalabel.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metalabel, f, indent=4)

    print(">>> [gen_meta_label.py] 处理完成")
    print(f"    扫描到的 txt 文件数: {total_txt_files}")
    print(f"    扫描到的行数: {total_lines}")
    print(f"    提取出的 utterance 数量: {len(metalabel)}")
    print(f"    metalabel.json 已保存到: {out_path.resolve()}")


if __name__ == "__main__":
    print(">>> [gen_meta_label.py] __name__ == '__main__'，准备调用 main()")
    main()
