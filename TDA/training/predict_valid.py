"""
DỰ ĐOÁN TRÊN TẬP VALID cho mọi mô hình đã train trên máy này (để chọn ngưỡng tin cậy cho F1 trên valid).
Không train gì. Dùng đúng thiết lập như khi chấm test (imgsz, conf 0.001, iou, max_det, dự đoán từng ảnh).

Chạy trong thư mục code (cạnh common.py, config.yaml, results):
    python predict_valid.py --data <thư mục split_data_by_box>

Kết quả:
    results\\<mô hình>\\seed<k>\\valid_predictions_coco.json   (mỗi lượt chạy)
    valid_gt_coco.json                                        (nhãn thật tập valid)
Lượt nào đã có file valid_predictions_coco.json thì bỏ qua (chạy lại an toàn khi bị ngắt).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import time
from pathlib import Path

import common


def find_runs(results_dir):
    runs = []
    for p in sorted(Path(results_dir).glob('*/seed*')):
        if p.parent.name.startswith('_'):
            continue                      # bỏ _smoke, _calib_*
        model, seed = p.parent.name, int(p.name[4:])
        w = p / ('best.pth' if model == common.FRCNN_KEY else 'train/weights/best.pt')
        if (p / 'metrics.json').exists() and w.exists():
            runs.append((model, seed, p, w))
    return runs


def predict_yolo(weights, items, path_to_id, y):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    preds = []
    for ip, _ in items:   # từng ảnh một, giống hệt lúc chấm test
        r = model.predict(source=str(ip), imgsz=y['imgsz'], conf=y['eval_conf'], iou=y['eval_iou'],
                          max_det=y['max_det'], device=0, verbose=False)[0]
        iid = path_to_id[str(Path(ip).resolve())]
        b = r.boxes
        for (x0, y0, x1, y1), c, s in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
            preds.append({'image_id': iid, 'category_id': int(c) + 1,
                          'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], 'score': float(s)})
    return preds


def predict_frcnn(weights, root, nc, f):
    import torch
    model = common.build_frcnn(nc, f).cuda()
    model.load_state_dict(torch.load(weights, map_location='cuda'))
    _, preds = common.frcnn_predict(model, root, 'valid', nc, f)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--allow-version-mismatch', action='store_true')
    args = ap.parse_args()
    cfg = common.load_config()
    common.check_versions(args.allow_version_mismatch)
    ds = common.check_dataset(args.data, cfg, require_fingerprint=True)
    print(f'config_sha256 = {cfg["_sha256"]}, dataset_fingerprint = {ds["fingerprint"]}')
    cfg['faster_rcnn']['workers'] = 0     # chỉ ảnh hưởng tốc độ đọc ảnh, không đổi kết quả; tránh thiếu RAM

    gt, path_to_id, items = common.build_coco_gt(ds['root'], 'valid', ds['nc'])
    common.save_json(gt, common.CODE_DIR / 'valid_gt_coco.json')
    print(f'Tập valid: {len(gt["images"])} ảnh, {len(gt["annotations"])} hộp → valid_gt_coco.json')

    runs = find_runs(common.CODE_DIR / cfg['run']['results_dir'])
    print('Các lượt sẽ dự đoán:', ', '.join(f'{m}/seed{s}' for m, s, _, _ in runs))
    for model, seed, run_dir, w in runs:
        out = run_dir / 'valid_predictions_coco.json'
        if out.exists():
            print(f'⏭  {model} seed {seed}: đã có, bỏ qua')
            continue
        t0 = time.time()
        common.seed_everything(0)
        if model == common.FRCNN_KEY:
            preds = predict_frcnn(w, ds['root'], ds['nc'], cfg['faster_rcnn'])
        else:
            preds = predict_yolo(w, items, path_to_id, cfg['yolo'])
        common.save_json(preds, out)
        print(f'✅ {model} seed {seed}: {len(preds)} dự đoán, {time.time() - t0:.0f} s → {out}')
    print('\nXong. Gửi lại: valid_gt_coco.json và các file results\\*\\seed*\\valid_predictions_coco.json')


if __name__ == '__main__':
    main()
