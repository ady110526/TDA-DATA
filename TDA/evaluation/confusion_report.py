"""
Ma trận nhầm lẫn từ các file cm_<mô hình>_seed<k>.npy do f1_valid_threshold.py tạo ra.

    python confusion_report.py --cm-dir f1_out --class-map class_group_mapping.csv \
        --models yolo26s frcnn_r50fpn --out cm_out

--class-map: bảng class_id → short, group do notebook 02_eda xuất ra (output_eda/tables/class_group_mapping.csv).
Các seed của cùng mô hình được cộng dồn. Đầu ra: hình ma trận theo 5 nhóm biển (chuẩn hóa theo hàng),
ma trận 115 lớp dạng CSV, bảng các cặp lớp bị nhầm nhiều nhất, tỉ lệ hộp bị gán sai lớp.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ['Cấm', 'Nguy hiểm', 'Hiệu lệnh', 'Chỉ dẫn', 'Phụ']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cm-dir', required=True)
    ap.add_argument('--class-map', required=True)
    ap.add_argument('--models', nargs='+', default=['yolo26s', 'frcnn_r50fpn'])
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--out', default='cm_out')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    mp = pd.read_csv(args.class_map, encoding='utf-8-sig').sort_values('class_id')
    short, grp = mp['short'].astype(str).tolist(), mp['group'].tolist()
    nc = len(short)
    gi = {g: i for i, g in enumerate(GROUPS)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(1, len(args.models), figsize=(5 * len(args.models), 4.3), constrained_layout=True,
                             squeeze=False)
    for ax, model in zip(axes[0], args.models):
        files = sorted(Path(args.cm_dir).glob(f'cm_{model}_seed*.npy'))
        cm = sum(np.load(f) for f in files)
        labels = short + ['nen']
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / f'cm_115_lop_{model}.csv', encoding='utf-8-sig')
        g = np.zeros((6, 6), int)
        for i in range(nc + 1):
            for j in range(nc + 1):
                g[gi[grp[i]] if i < nc else 5, gi[grp[j]] if j < nc else 5] += cm[i, j]
        g = g[:5]
        n = g / np.maximum(g.sum(1, keepdims=True), 1) * 100
        pd.DataFrame(n.round(2), index=GROUPS, columns=GROUPS + ['Nền (bỏ sót)']).to_csv(
            out / f'cm_5_nhom_{model}.csv', encoding='utf-8-sig')
        pairs = sorted(((cm[i, j], short[i], short[j]) for i in range(nc) for j in range(nc)
                        if i != j and cm[i, j] > 0), reverse=True)
        pd.DataFrame(pairs[:args.top], columns=['so_hop', 'lop_that', 'bi_nhan_thanh']).to_csv(
            out / f'cap_nham_nhieu_nhat_{model}.csv', index=False, encoding='utf-8-sig')
        matched = cm[:nc, :nc].sum()
        print(f'{model} ({len(files)} seed): {sum(p[0] for p in pairs)}/{matched} hộp được ghép bị gán sai lớp '
              f'({sum(p[0] for p in pairs) / max(matched, 1) * 100:.1f}%)')
        im = ax.imshow(n, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks(range(6))
        ax.set_xticklabels(['Cấm', 'Nguy\nhiểm', 'Hiệu\nlệnh', 'Chỉ\ndẫn', 'Phụ', 'Nền\n(bỏ sót)'])
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'{x} (n={g[i].sum()})' for i, x in enumerate(GROUPS)])
        for i in range(5):
            for j in range(6):
                if n[i, j] >= 0.05:
                    ax.text(j, i, f'{n[i, j]:.1f}'.replace('.', ','), ha='center', va='center', fontsize=8,
                            color='white' if n[i, j] > 60 else '#1a1a1a')
        ax.set_title(f'{model} ({len(files)} seed)')
        ax.set_xlabel('Nhóm dự đoán')
        ax.set_ylabel('Nhóm thật')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    cb.set_label('% số hộp thật của nhóm')
    cb.outline.set_visible(False)
    fig.savefig(out / 'confusion_5_nhom.png', dpi=200)
    print(f'Đã lưu vào {out.resolve()}')


if __name__ == '__main__':
    main()
