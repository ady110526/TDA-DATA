"""
Phân tích AP@[.5:.95] theo lớp (Mục "Kết quả theo lớp" trong bài).

    python per_class_ap.py --results results_may1 results_may2 results_may3 \
        --class-dist class_distribution.csv --test-gt test_gt_coco.json --out per_class_ap.csv

--class-dist: output_eda/tables/class_distribution.csv của notebook 02_eda.
In ra: tương quan hạng Spearman giữa các mô hình, tương quan AP–số hộp, AP theo mức tần suất và theo nhóm,
và các lớp khó nhất. Lưu bảng AP từng lớp (trung bình các seed) cho từng mô hình.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ORDER = ['yolov8n', 'yolo11n', 'yolo26n', 'yolov8s', 'yolo11s', 'yolo26s', 'frcnn_r50fpn']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--class-dist', required=True)
    ap.add_argument('--test-gt', required=True)
    ap.add_argument('--out', default='per_class_ap.csv')
    args = ap.parse_args()
    rows = []
    for d in args.results:
        for f in Path(d).glob('*/seed*/per_class_AP_test.csv'):
            if f.parts[-3].startswith('_'):
                continue
            t = pd.read_csv(f, encoding='utf-8-sig')
            t['model'] = f.parts[-3]
            rows.append(t)
    a = pd.concat(rows)
    P = a.pivot_table(index='class_id', columns='model', values='AP50_95', aggfunc='mean')
    models = [m for m in ORDER if m in P.columns] + [m for m in P.columns if m not in ORDER]
    P = P[models]
    cd = pd.read_csv(args.class_dist, encoding='utf-8-sig').set_index('class_id')
    gt = json.load(open(args.test_gt, encoding='utf-8'))
    P['avg'] = P[models].mean(axis=1)
    P['short'] = cd['short'].astype(str)
    P['group'] = cd['group']
    P['n_boxes'] = cd['n_boxes']
    P['tier'] = cd['frequency_tier']
    P['n_test'] = pd.Series([x['category_id'] - 1 for x in gt['annotations']]).value_counts()
    P.to_csv(args.out, encoding='utf-8-sig')
    pair = [spearmanr(P[x], P[y]).statistic for i, x in enumerate(models) for y in models[i + 1:]]
    print(f'Spearman giữa các cặp mô hình: {min(pair):.2f} – {max(pair):.2f}')
    print(f'Spearman AP trung bình – số hộp: {spearmanr(P["avg"], P["n_boxes"]).statistic:.2f}')
    print('AP trung bình theo mức tần suất:\n', P.groupby('tier')['avg'].mean().round(3).to_string())
    print('AP trung bình theo nhóm:\n', P.groupby('group')['avg'].mean().round(3).to_string())
    print(f'Số lớp AP ≥ 0,80: {(P["avg"] >= 0.8).sum()}; AP < 0,50: {(P["avg"] < 0.5).sum()}')
    print('Các lớp khó nhất:\n', P.sort_values('avg')[['short', 'group', 'n_test'] + models + ['avg']]
          .head(10).round(2).to_string())


if __name__ == '__main__':
    main()
