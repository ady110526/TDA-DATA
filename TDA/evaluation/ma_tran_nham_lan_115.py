"""
Vẽ ma trận nhầm lẫn 115 lớp của MỘT lần huấn luyện (Hình ma trận nhầm lẫn trong bài: YOLO26s seed 1).

    python ma_tran_nham_lan_115.py --cm f1_out/cm_yolo26s_seed1.npy \
        --class-map class_group_mapping.csv --mode phan_tram --out 16_confusion_115_yolo26s_seed1.png

--cm        : file cm_<mô hình>_seed<k>.npy do f1_valid_threshold.py tạo (ngưỡng chọn trên valid, IoU >= 0,5,
              ghép không phân biệt lớp; hàng/cột cuối = "nền").
--class-map : output_eda/tables/class_group_mapping.csv (notebook 02_eda).
--mode      : phan_tram (bản dùng trong bài) – màu và số đỏ = % số hộp thật của lớp, ghi số ở ô ngoài đường chéo >= 10%;
              so_hop – ghi số hộp ở ô >= 10, có thêm hàng "nền" (dự đoán không khớp hộp thật nào).
Lớp được xếp theo 5 nhóm biển; đường đỏ là ranh giới nhóm.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

GROUPS = ['Cấm', 'Nguy hiểm', 'Hiệu lệnh', 'Chỉ dẫn', 'Phụ']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cm', required=True)
    ap.add_argument('--class-map', required=True)
    ap.add_argument('--mode', choices=['phan_tram', 'so_hop'], default='phan_tram')
    ap.add_argument('--nguong', type=float, default=10, help='ngưỡng ghi số (%% hoặc số hộp), mặc định 10')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    mp = pd.read_csv(args.class_map, encoding='utf-8-sig').sort_values('class_id')
    S, grp = mp['short'].astype(str).tolist(), mp['group'].tolist()
    nc = len(S)
    cm = np.load(args.cm)
    assert cm.shape == (nc + 1, nc + 1), f'ma trận {cm.shape} không khớp {nc} lớp'
    order = sorted(range(nc), key=lambda i: (GROUPS.index(grp[i]), i))
    idx = order + [nc]
    M = cm[np.ix_(idx, idx)]
    lab = [S[i] for i in order] + ['nền']
    count = args.mode == 'so_hop'
    R = nc + 1 if count else nc
    P = np.zeros((R, nc + 1))
    P[:nc] = 100 * M[:nc] / np.clip(M[:nc].sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(19, 18.5 if count else 18.3))
    mask = np.zeros_like(P, bool)
    if count:
        mask[nc] = True
    im1 = ax.imshow(np.ma.masked_where(mask, P), cmap='Blues', norm=PowerNorm(0.4, 0, 100),
                    interpolation='nearest', aspect='auto')
    if count:
        fa = np.tile(M[nc].astype(float), (R, 1))
        ax.imshow(np.ma.masked_where(~mask, fa), cmap='Greys', vmin=0, vmax=max(fa.max(), 1) * 1.6,
                  interpolation='nearest', aspect='auto')
    for i in range(R):
        for j in range(nc + 1):
            if i == j:
                continue
            v = M[i, j] if count else P[i, j]
            if v >= args.nguong:
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7.5, color='#c00000', fontweight='bold')
    b = [0]
    for g in GROUPS:
        b.append(b[-1] + sum(grp[i] == g for i in order))
    for x in b[1:]:
        ax.axhline(x - 0.5, color='#d03030', lw=0.8)
        ax.axvline(x - 0.5, color='#d03030', lw=0.8)
    for k, g in enumerate(GROUPS):
        ax.text(nc + 1.2, (b[k] + b[k + 1]) / 2 - 0.5, g, rotation=-90, va='center', fontsize=11)
    ax.set_xticks(range(nc + 1)); ax.set_xticklabels(lab, rotation=90, fontsize=6.5)
    ax.set_yticks(range(R)); ax.set_yticklabels(lab[:R], fontsize=6.5)
    ax.tick_params(length=0)
    ax.set_ylabel('Lớp thật', fontsize=12)
    ax.set_xlabel('Lớp dự đoán (cột cuối "nền": hộp thật bị bỏ sót)', fontsize=12)
    cb = fig.colorbar(im1, ax=ax, shrink=0.35, pad=0.04)
    cb.set_label('% số hộp thật của lớp (thang lũy thừa 0,4)', fontsize=11)
    plt.tight_layout()
    fig.savefig(args.out, dpi=200)
    print('Đã lưu', args.out)


if __name__ == '__main__':
    main()
