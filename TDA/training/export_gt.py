"""
Xuất nhãn thật (ground truth) của một tập (test hoặc valid) sang COCO JSON, đúng định dạng mà
common.py dùng khi chấm mAP. Dùng làm đầu vào cho các script trong thư mục evaluation/.
Không train, không cần GPU.

    python export_gt.py --data <thư mục split_data_by_box> --split test
    python export_gt.py --data <thư mục split_data_by_box> --split valid
Kết quả: <split>_gt_coco.json (test: 2.206 ảnh / 4.073 hộp; valid: 2.203 ảnh / 4.068 hộp)
"""
import argparse

import common

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--split', choices=['test', 'valid'], default='test')
args = ap.parse_args()
cfg = common.load_config()
ds = common.check_dataset(args.data, cfg, require_fingerprint=True)
gt, _, _ = common.build_coco_gt(ds['root'], args.split, ds['nc'])
gt['class_names'] = ds['names']
out = f'{args.split}_gt_coco.json'
common.save_json(gt, out)
print(f'✅ {out}: {len(gt["images"])} ảnh, {len(gt["annotations"])} hộp, {ds["nc"]} lớp')
