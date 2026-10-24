"""
Kiểm tra các "điểm nóng" trên bản đồ vị trí tâm hộp có phải do ảnh gần trùng hay không (Mục Phân tích dữ liệu).

    python heatmap_group_check.py --boxes boxes.csv --groups image_group_split_both.csv

--boxes: output_eda/tables/boxes.csv (notebook 02_eda); --groups: output_split/tables/image_group_split_both.csv
(notebook 03_group_split). Chia khung ảnh thành lưới N×N, chọn các ô vượt trội nhất so với trung vị vùng
lân cận, rồi xem một nhóm ảnh gần trùng chiếm bao nhiêu phần trăm số hộp của ô.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boxes', required=True)
    ap.add_argument('--groups', required=True)
    ap.add_argument('--grid', type=int, default=50)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--top', type=int, default=10)
    args = ap.parse_args()
    b = pd.read_csv(args.boxes, encoding='utf-8-sig')
    g = pd.read_csv(args.groups, encoding='utf-8-sig')
    b = b.merge(g[['image', 'group_id', 'group_size']], on='image', how='left')
    N = args.grid
    b['cx'] = np.clip((b['xc'] * N).astype(int), 0, N - 1)
    b['cy'] = np.clip((b['yc'] * N).astype(int), 0, N - 1)
    H = np.zeros((N, N))
    for (y, x), c in b.groupby(['cy', 'cx']).size().items():
        H[y, x] = c
    M = median_filter(H, size=args.window)
    order = np.dstack(np.unravel_index(np.argsort(-(H - M).ravel()), H.shape))[0][:args.top]
    print(f'Toàn bộ dữ liệu: {len(b)} hộp, trung bình {len(b) / b["group_id"].nunique():.2f} hộp mỗi nhóm')
    for y, x in order:
        s = b[(b['cy'] == y) & (b['cx'] == x)]
        top = s['group_id'].value_counts()
        print(f'ô (hàng {y}, cột {x}): {int(H[y, x])} hộp (trung vị lân cận {M[y, x]:.0f}), '
              f'{s["group_id"].nunique()} nhóm; nhóm lớn nhất chiếm {top.iloc[0]} hộp ({top.iloc[0] / len(s) * 100:.0f}%)')


if __name__ == '__main__':
    main()
