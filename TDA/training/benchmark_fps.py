"""
ĐO TỐC ĐỘ SUY LUẬN (FPS) CHO CẢ 7 MÔ HÌNH – CHẠY TRÊN MỘT MÁY DUY NHẤT

Chạy SAU KHI 3 máy đã train xong và đã gom thư mục results về máy đo (đề xuất máy 1, RTX 5050).
Mọi mô hình được đo trong cùng điều kiện: cùng GPU, batch = 1, cùng 300 ảnh test, có làm nóng GPU.

Cách chạy (terminal VS Code, môi trường chung):
    python benchmark_fps.py --data D:\\VNTrafficSign\\split_data_by_box --results results_may1 results_may2 results_may3

Kết quả: fps\\fps_results.csv và fps\\env.json
Đo cho cả FP32 và FP16. Thời gian tính từ lúc ảnh đã nằm trong RAM (không tính đọc file),
gồm: tiền xử lý + chạy mô hình + hậu xử lý (NMS). Đồng bộ GPU trước khi bấm giờ.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import common

N_WARMUP = 50
N_IMAGES = 300
SEED_FOR_WEIGHTS = 0          # đo trọng số của seed 0 cho mọi mô hình


def find_weights(results_dirs, model_key):
    name = 'best.pth' if model_key == common.FRCNN_KEY else 'best.pt'
    for d in results_dirs:
        for p in Path(d).rglob(name):
            parts = p.parts
            if '_smoke' in parts or any(x.startswith('_calib') for x in parts):
                continue
            if model_key in parts and f'seed{SEED_FOR_WEIGHTS}' in parts:
                return p
    return None


def summarize(times_ms):
    t = np.array(times_ms)
    return {'ms_mean': round(float(t.mean()), 3), 'ms_median': round(float(np.median(t)), 3),
            'ms_p95': round(float(np.percentile(t, 95)), 3), 'fps_mean': round(1000.0 / float(t.mean()), 2)}


def bench_yolo(weights, images_bgr, imgsz, half):
    import torch
    from ultralytics import YOLO
    import logging
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)   # tắt cảnh báo in ra màn hình trong lúc bấm giờ (in ra làm chậm kết quả đo)
    model = YOLO(str(weights))
    kw = dict(imgsz=imgsz, conf=0.25, iou=0.7, device=0, verbose=False)
    if half:
        kw['half'] = True            # FP32 thì không truyền 'half' (tránh cảnh báo deprecated)
    for im in images_bgr[:N_WARMUP]:
        model.predict(im, **kw)
    fp16_active = bool(getattr(getattr(model.predictor, 'model', None), 'fp16', False))
    times = []
    for im in images_bgr:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(im, **kw)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    n_params = sum(p.numel() for p in model.model.parameters())
    return times, n_params, fp16_active


def bench_frcnn(weights, images_bgr, cfg_f, nc, half):
    import cv2
    import torch
    model = common.build_frcnn(nc, {**cfg_f, 'eval_score_thresh': 0.25}).cuda().eval()
    model.load_state_dict(torch.load(weights, map_location='cuda'))

    def run(im):
        x = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).cuda().float().div_(255)
        with torch.inference_mode(), torch.autocast('cuda', enabled=half):
            return model([x])

    for im in images_bgr[:N_WARMUP]:
        run(im)
    times = []
    for im in images_bgr:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(im)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    n_params = sum(p.numel() for p in model.parameters())
    return times, n_params, half


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--results', nargs='+', required=True, help='Các thư mục results của 3 máy')
    ap.add_argument('--allow-version-mismatch', action='store_true')
    args = ap.parse_args()

    cfg = common.load_config()
    common.check_versions(args.allow_version_mismatch)
    env = common.env_info()
    ds = common.check_dataset(args.data, cfg, require_fingerprint=True)
    print(f'Đo FPS trên {env["gpu_name"]} ({env["vram_gb"]} GB), driver {env["driver"]}')
    print('Trong lúc đo: cắm sạc, chế độ hiệu năng cao, đóng mọi ứng dụng khác.')

    items = common.list_split(ds['root'], 'test')
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(items), size=min(N_IMAGES, len(items)), replace=False))
    images = [cv2.imread(str(items[i][0]), cv2.IMREAD_COLOR) for i in pick]
    print(f'Dùng {len(images)} ảnh test (cố định), làm nóng {N_WARMUP} ảnh.')

    models = ['yolov8n', 'yolo11n', 'yolo26n', 'yolov8s', 'yolo11s', 'yolo26s', common.FRCNN_KEY]
    out_dir = common.CODE_DIR / 'fps'
    out_dir.mkdir(exist_ok=True)
    common.save_json(env, out_dir / 'env.json')
    rows = []
    for m in models:
        w = find_weights(args.results, m)
        if w is None:
            print(f'⚠️ Không tìm thấy trọng số {m} seed {SEED_FOR_WEIGHTS}, bỏ qua')
            continue
        for half in (False, True):
            common.seed_everything(0)
            if m == common.FRCNN_KEY:
                times, n_params, fp16_active = bench_frcnn(w, images, cfg['faster_rcnn'], ds['nc'], half)
                inp = f"{cfg['faster_rcnn']['min_size']}-{cfg['faster_rcnn']['max_size']}"
            else:
                times, n_params, fp16_active = bench_yolo(w, images, cfg['yolo']['imgsz'], half)
                inp = str(cfg['yolo']['imgsz'])
            s = summarize(times)
            row = {'model': m, 'precision': 'FP16' if half else 'FP32', 'input': inp, 'batch': 1,
                   'fp16_active': fp16_active, 'params_M': round(n_params / 1e6, 2), 'weights_MB': round(w.stat().st_size / 1e6, 1), **s,
                   'gpu': env['gpu_name'], 'weights': str(w)}
            rows.append(row)
            print(f"{m:14s} {row['precision']}: {s['ms_mean']:.2f} ms/ảnh ({s['fps_mean']:.1f} FPS), p95 {s['ms_p95']:.2f} ms, FP16 thật sự bật: {fp16_active}")
    with open(out_dir / 'fps_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n✅ Đã lưu {out_dir / "fps_results.csv"}')


if __name__ == '__main__':
    main()
