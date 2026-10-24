"""
F1 với ngưỡng tin cậy CHỌN TRÊN TẬP VALID (đúng như bài báo), kèm F1 ở ngưỡng cố định để tham khảo.

Với mỗi lần huấn luyện (thư mục results/<mô hình>/seed<k> có cả valid_predictions_coco.json và
test_predictions_coco.json): dò ngưỡng trên valid, lấy ngưỡng cho F1 vi mô cao nhất, áp nguyên ngưỡng đó lên test.

    python f1_valid_threshold.py --valid-gt valid_gt_coco.json --test-gt test_gt_coco.json \
        --results results_may1 results_may2 results_may3 --out f1_out

Đầu ra (thư mục --out): f1_runs.csv (từng lần chạy), f1_summary.csv (trung bình ± độ lệch chuẩn theo mô hình),
cm_<mô hình>_seed<k>.npy (ma trận nhầm lẫn trên test ở ngưỡng chọn trên valid), f1_per_class_<mô hình>_seed<k>.npy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detmetrics import evaluate, load_gt, load_preds, summarize


def find_runs(dirs):
    runs = []
    for d in dirs:
        for p in sorted(Path(d).glob('*/seed*')):
            if p.parent.name.startswith('_'):
                continue                       # bỏ _smoke, _calib_*
            v, t = p / 'valid_predictions_coco.json', p / 'test_predictions_coco.json'
            if v.exists() and t.exists():
                runs.append((p.parent.name, int(p.name[4:]), v, t))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--valid-gt', required=True)
    ap.add_argument('--test-gt', required=True)
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--out', default='f1_out')
    ap.add_argument('--grid', nargs=3, type=float, default=[0.05, 0.95, 0.01], metavar=('MIN', 'MAX', 'STEP'))
    ap.add_argument('--fixed-conf', type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    vids, vgt, nc, _ = load_gt(args.valid_gt)
    tids, tgt, nc2, _ = load_gt(args.test_gt)
    assert nc == nc2
    grid = np.round(np.arange(args.grid[0], args.grid[1] + 1e-9, args.grid[2]), 4)
    rows = []
    for model, seed, vf, tf in find_runs(args.results):
        vp, tp_ = load_preds(vf), load_preds(tf)
        best_f, best_c = -1.0, None
        for c in grid:
            d, _ = summarize(*evaluate(vp, vids, vgt, nc, c)[:3])
            if d['F1_micro'] > best_f:
                best_f, best_c = d['F1_micro'], float(c)
        a, b, c_, cm = evaluate(tp_, tids, tgt, nc, best_c)
        dsel, fc = summarize(a, b, c_)
        dfix, _ = summarize(*evaluate(tp_, tids, tgt, nc, args.fixed_conf)[:3])
        np.save(out / f'cm_{model}_seed{seed}.npy', cm)
        np.save(out / f'f1_per_class_{model}_seed{seed}.npy', fc)
        rows.append({'model': model, 'seed': seed, 'conf_valid': best_c, 'F1_valid': best_f,
                     **{f'{k}_test': v for k, v in dsel.items()},
                     f'F1_micro_test_conf{args.fixed_conf}': dfix['F1_micro']})
        print(f'{model:14s} seed {seed}: ngưỡng {best_c:.2f} | F1 test {dsel["F1_micro"]:.4f} '
              f'| F1@{args.fixed_conf} {dfix["F1_micro"]:.4f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'f1_runs.csv', index=False)
    cols = [c for c in df.columns if c not in ('model', 'seed')]
    df.groupby('model')[cols].agg(['mean', 'std']).to_csv(out / 'f1_summary.csv')
    print(f'\nĐã lưu vào {out.resolve()}')


if __name__ == '__main__':
    main()
