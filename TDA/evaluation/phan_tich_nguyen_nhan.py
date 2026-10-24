"""
Các con số dùng để giải thích ma trận nhầm lẫn trong bài (mục "F1 và ma trận nhầm lẫn"), cho MỘT lần huấn luyện.

    python phan_tich_nguyen_nhan.py --gt test_gt_coco.json \
        --pred results_may1/yolo26s/seed1/test_predictions_coco.json --conf 0.37 \
        --class-map class_group_mapping.csv

Cách ghép giống hệt ma trận nhầm lẫn (dùng lại hàm của training/xuat_anh_nham_lan.py): dự đoán có điểm >= conf,
IoU >= 0,5, không phân biệt lớp, theo điểm tin cậy giảm dần. In ra:
  - số hộp đúng lớp / nhầm lớp / bỏ sót, số dự đoán báo nhầm, số ca nhầm lớp khác nhóm biển;
  - chẩn đoán tự động hộp bị bỏ sót (điểm thấp / không phát hiện / lệch hộp);
  - kích thước hộp (căn bậc hai diện tích): toàn tập, hộp bị bỏ sót, hộp bị nhầm lớp; tỉ lệ hộp nhỏ (< 32^2);
  - các lớp bị bỏ sót nhiều nhất và kích thước hộp bị bỏ sót so với trung vị của lớp;
  - các cặp nhầm nhiều nhất và số ca theo chiều ngược lại.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from xuat_anh_nham_lan import match_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--conf', type=float, required=True)
    ap.add_argument('--class-map', required=True)
    ap.add_argument('--top', type=int, default=7)
    args = ap.parse_args()
    mp = pd.read_csv(args.class_map, encoding='utf-8-sig').sort_values('class_id')
    S, grp = mp['short'].astype(str).tolist(), mp['group'].tolist()
    nc = len(S)
    gt = json.load(open(args.gt, encoding='utf-8'))
    preds = json.load(open(args.pred, encoding='utf-8'))
    cm, ev = match_all(gt, preds, args.conf, nc)
    side = lambda b: float(np.sqrt(b[2] * b[3]))

    off = cm[:nc, :nc].sum() - np.trace(cm[:nc, :nc])
    matched = cm[:nc, :nc].sum()
    cross = sum(int(cm[i, j]) for i in range(nc) for j in range(nc) if i != j and grp[i] != grp[j])
    print(f'Hộp thật: {len(gt["annotations"])} | đúng lớp {int(np.trace(cm[:nc, :nc]))} | nhầm lớp {int(off)} '
          f'({100 * off / matched:.1f}% số hộp được ghép) | bỏ sót {int(cm[:nc, nc].sum())} | báo nhầm {int(cm[nc, :nc].sum())}')
    print(f'Nhầm lớp sang nhóm biển khác: {cross}')
    c = collections.Counter(e[8].split(':')[0] for e in ev if e[0] == 'bo_sot')
    print('Chẩn đoán tự động hộp bị bỏ sót:', dict(c))

    allb = np.array([side(a['bbox']) for a in gt['annotations']])
    mis = np.array([side(e[3]) for e in ev if e[0] == 'bo_sot'])
    con = np.array([side(e[3]) for e in ev if e[0] == 'nham_lop'])
    for name, x in (('toàn tập test', allb), ('bị bỏ sót', mis), ('bị nhầm lớp', con)):
        print(f'Hộp {name:14s}: n={len(x):5d}  cạnh trung vị {np.median(x):.1f} px  hộp nhỏ (<32^2) {100 * (x < 32).mean():.1f}%')

    print(f'\n{args.top} lớp bị bỏ sót nhiều nhất (cạnh trung vị hộp bị bỏ sót / của cả lớp):')
    for i in np.argsort(-cm[:nc, nc])[:args.top]:
        xs = [side(e[3]) for e in ev if e[0] == 'bo_sot' and e[2] == i]
        al = [side(a['bbox']) for a in gt['annotations'] if a['category_id'] - 1 == i]
        print(f'  {S[i]:8s} bỏ sót {int(cm[i, nc]):3d}/{int(cm[i].sum()):4d}   {np.median(xs):.1f} / {np.median(al):.1f} px')

    print('\nCác cặp nhầm nhiều nhất (A → B, chiều ngược B → A):')
    pairs = sorted(((int(cm[i, j]), i, j) for i in range(nc) for j in range(nc) if i != j and cm[i, j] > 0), reverse=True)
    for n, i, j in pairs[:10]:
        print(f'  {S[i]:8s} → {S[j]:8s} {n:3d}   (ngược lại {int(cm[j, i])}; số hộp test {int(cm[i].sum())} / {int(cm[j].sum())})')


if __name__ == '__main__':
    main()
