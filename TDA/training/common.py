"""
Phần dùng chung cho 3 file train (VN Traffic Sign benchmark).

- Kiểm tra máy: phiên bản thư viện, GPU, mã cấu hình, dấu vân tay dữ liệu
- Train + đánh giá YOLO (Ultralytics) và Faster R-CNN ResNet50-FPN (torchvision)
- Đánh giá CHUNG cho mọi mô hình bằng pycocotools trên tập test (cùng một cách tính)
- Ghi kết quả + môi trường của từng lần chạy, tổng hợp mean ± std

Không sửa file này riêng trên một máy. Mọi máy dùng cùng một bản.
Tổng hợp kết quả từ nhiều máy:  python common.py aggregate <thư mục results 1> <thư mục results 2> ...
"""
import os

# Phải đặt TRƯỚC khi import torch để cuBLAS chạy tất định
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

CODE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CODE_DIR / 'config.yaml'
SPLITS = ('train', 'valid', 'test')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# Phiên bản đã khóa cho cả nhóm (xem requirements.txt)
PINNED = {'torch': '2.11.0', 'torchvision': '0.26.0', 'ultralytics': '8.4.178', 'pycocotools': '2.0.11'}

YOLO_WEIGHTS = {
    'yolov8n': 'yolov8n.pt', 'yolov8s': 'yolov8s.pt',
    'yolo11n': 'yolo11n.pt', 'yolo11s': 'yolo11s.pt',
    'yolo26n': 'yolo26n.pt', 'yolo26s': 'yolo26s.pt',
}
FRCNN_KEY = 'frcnn_r50fpn'
CALIB_MODEL, CALIB_SEED = 'yolov8n', 0   # lần chạy hiệu chuẩn: giống nhau trên cả 3 máy


# ----------------------------------------------------------------------------------
# Cấu hình, phiên bản, môi trường
# ----------------------------------------------------------------------------------
def load_config():
    raw = CONFIG_PATH.read_bytes()
    cfg = yaml.safe_load(raw.decode('utf-8'))
    cfg['_sha256'] = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()[:16]
    return cfg


def installed_versions():
    from importlib import metadata
    out = {}
    for pkg in PINNED:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = 'MISSING'
    return out


def check_versions(allow_mismatch=False):
    vers = installed_versions()
    bad = {k: v for k, v in vers.items() if not str(v).split('+')[0] == PINNED[k]}
    if bad:
        msg = '\n'.join(f'  {k}: cài {v}, cần {PINNED[k]}' for k, v in bad.items())
        if allow_mismatch:
            print(f'⚠️ Phiên bản khác bản đã khóa (đang bỏ qua vì --allow-version-mismatch):\n{msg}')
        else:
            raise SystemExit(f'❌ Phiên bản thư viện khác bản đã khóa cho cả nhóm:\n{msg}\n'
                             'Cài lại theo requirements.txt (xem HUONG_DAN.md).')
    return vers


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('❌ Không thấy GPU CUDA. Kiểm tra driver NVIDIA và bản PyTorch cu128.')
    p = torch.cuda.get_device_properties(0)
    driver = ''
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    return {'gpu_name': p.name, 'vram_gb': round(p.total_memory / 1024 ** 3, 2),
            'compute_capability': f'{p.major}.{p.minor}', 'driver': driver,
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}


def env_info():
    import torch
    return {'python': sys.version.split()[0], 'os': platform.platform(), 'host': socket.gethostname(),
            **installed_versions(), **gpu_info(), 'torch_num_threads': torch.get_num_threads(),
            'time': datetime.now().isoformat(timespec='seconds')}


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------------
def dataset_fingerprint(root):
    """Băm tên + kích thước mọi file, và nội dung file nhãn/cấu hình. Giống nhau ⇔ cùng bộ dữ liệu."""
    h = hashlib.sha256()
    root = Path(root)
    junk = {'thumbs.db', 'desktop.ini', '.ds_store'}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name.lower() in junk or p.name.startswith('.') or p.suffix.lower() == '.cache':
            continue   # bỏ file rác của hệ điều hành và file cache nhãn mà Ultralytics tự tạo
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode('utf-8'))
        h.update(str(p.stat().st_size).encode())
        if p.suffix.lower() in {'.txt', '.yaml', '.csv'}:
            h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:16]


def detect_split_mode(root):
    readme = Path(root) / 'README_split.md'
    if not readme.exists():
        return None
    t = readme.read_text(encoding='utf-8')
    if '**số box**' in t:
        return 'split_data_by_box'
    if '**số ảnh chứa**' in t:
        return 'split_data_by_image'
    return None


def check_dataset(root, cfg, require_fingerprint=True):
    root = Path(root).resolve()
    for s in SPLITS:
        for sub in ('images', 'labels'):
            if not (root / s / sub).is_dir():
                raise SystemExit(f'❌ Thiếu thư mục {root / s / sub}. --data phải trỏ tới thư mục đã giải nén split_data_*.zip')
    data_yaml = yaml.safe_load(open(root / 'data.yaml', encoding='utf-8'))
    names = list(data_yaml['names'].values()) if isinstance(data_yaml['names'], dict) else list(data_yaml['names'])
    if int(data_yaml['nc']) != len(names):
        raise SystemExit('❌ data.yaml: nc khác số tên class')
    mode = detect_split_mode(root)
    want = cfg['dataset']['name']
    if not want:
        if require_fingerprint:
            raise SystemExit('❌ config.yaml: dataset.name đang trống. Nhóm chọn split_data_by_box hoặc split_data_by_image rồi điền vào.')
        print('⚠️ config.yaml: dataset.name đang trống (bắt buộc điền trước khi train thật).')
    elif mode and mode != want:
        raise SystemExit(f'❌ Thư mục dữ liệu là {mode} nhưng config.yaml ghi {want}.')
    fp = dataset_fingerprint(root)
    want_fp = cfg['dataset']['fingerprint']
    if want_fp and fp != want_fp:
        raise SystemExit(f'❌ Dấu vân tay dữ liệu {fp} khác config ({want_fp}). Máy này đang dùng bộ dữ liệu khác!')
    if require_fingerprint and not want_fp:
        raise SystemExit(f'❌ config.yaml: dataset.fingerprint đang trống. Dán giá trị này vào config rồi chép config sang 2 máy kia:\n'
                         f'    fingerprint: "{fp}"')
    return {'root': str(root), 'names': names, 'nc': len(names), 'fingerprint': fp, 'mode': mode}


def resolved_data_yaml(ds, out_dir):
    """Ghi data.yaml có đường dẫn tuyệt đối (tránh lỗi đường dẫn tương đối trên Windows)."""
    out = Path(out_dir) / 'data_resolved.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {'path': ds['root'], 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
           'nc': ds['nc'], 'names': ds['names']}
    with open(out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return str(out)


def list_split(root, split):
    img_dir, lbl_dir = Path(root) / split / 'images', Path(root) / split / 'labels'
    items = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            items.append((p, lbl_dir / f'{p.stem}.txt'))
    return items


def exif_size(path):
    """Kích thước (w, h) sau khi xoay theo EXIF, giống cách Ultralytics và OpenCV đọc ảnh."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        try:
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                w, h = h, w
        except Exception:
            pass
    return w, h


def read_yolo_label(path):
    """Trả về list (class_id, xc, yc, w, h) chuẩn hóa. Notebook 03 đã đổi polygon → bbox."""
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) == 5:
            c, xc, yc, w, h = parts
            out.append((int(float(c)), float(xc), float(yc), float(w), float(h)))
    return out


def build_coco_gt(root, split, nc):
    items = list_split(root, split)
    images, anns, path_to_id = [], [], {}
    aid = 1
    for iid, (ip, lp) in enumerate(items, 1):
        W, H = exif_size(ip)
        images.append({'id': iid, 'file_name': ip.name, 'width': W, 'height': H})
        path_to_id[str(ip.resolve())] = iid
        for c, xc, yc, w, h in read_yolo_label(lp):
            bw, bh = w * W, h * H
            x0, y0 = xc * W - bw / 2, yc * H - bh / 2
            anns.append({'id': aid, 'image_id': iid, 'category_id': c + 1, 'bbox': [x0, y0, bw, bh],
                         'area': bw * bh, 'iscrowd': 0})
            aid += 1
    cats = [{'id': k + 1, 'name': str(k)} for k in range(nc)]
    return {'images': images, 'annotations': anns, 'categories': cats}, path_to_id, items


def coco_evaluate(gt_dict, preds, names):
    """mAP kiểu COCO (pycocotools). preds: list dict image_id, category_id, bbox[x,y,w,h], score."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io
    gt = COCO()
    gt.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    keys = ['mAP50_95', 'mAP50', 'mAP75', 'AP_small', 'AP_medium', 'AP_large']
    if not preds:
        return {k: 0.0 for k in keys}, [0.0] * len(names)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    res = {k: float(ev.stats[i]) for i, k in enumerate(keys)}
    prec = ev.eval['precision']            # [T, R, K, A, M]
    per_class = []
    for k in range(prec.shape[2]):
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        per_class.append(float(p.mean()) if p.size else float('nan'))
    return res, per_class


# ----------------------------------------------------------------------------------
# Lưu kết quả
# ----------------------------------------------------------------------------------
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_summary(results_dir, row):
    import csv
    path = Path(results_dir) / 'summary.csv'
    new = not path.exists()
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def finish_run(run_dir, ctx, model_key, seed, coco_res, per_class, extra):
    names = ctx['ds']['names']
    metrics = {'model': model_key, 'seed': seed, 'machine': ctx['machine']['id'],
               'gpu': ctx['env']['gpu_name'], 'config_sha256': ctx['cfg']['_sha256'],
               'dataset_fingerprint': ctx['ds']['fingerprint'], 'dataset': ctx['cfg']['dataset']['name'],
               'coco_test': coco_res, **extra}
    save_json(metrics, run_dir / 'metrics.json')
    import csv
    with open(run_dir / 'per_class_AP_test.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'class_name', 'AP50_95'])
        for i, (n, ap) in enumerate(zip(names, per_class)):
            w.writerow([i, n, ap])
    row = {'model': model_key, 'seed': seed, 'machine': ctx['machine']['id'], 'gpu': ctx['env']['gpu_name'],
           **{f'coco_{k}': round(v, 5) for k, v in coco_res.items()},
           **{k: v for k, v in extra.items() if not isinstance(v, (dict, list))},
           'config_sha256': ctx['cfg']['_sha256'], 'dataset_fingerprint': ctx['ds']['fingerprint']}
    append_summary(ctx['results_dir'], row)
    print(f'✅ {model_key} seed {seed}: COCO test mAP50-95 = {coco_res["mAP50_95"]:.4f}, mAP50 = {coco_res["mAP50"]:.4f}')


# ----------------------------------------------------------------------------------
# YOLO (Ultralytics)
# ----------------------------------------------------------------------------------
def _set_resume_workers(last, n):
    """Ultralytics lấy 'workers' từ checkpoint khi resume → ghi đè số workers vào last.pt."""
    import torch
    ck = torch.load(str(last), map_location='cpu', weights_only=False)
    ta = ck.get('train_args')
    if isinstance(ta, dict) and ta.get('workers') != n:
        ta['workers'] = n
        tmp = Path(str(last) + '.tmp')
        torch.save(ck, str(tmp))
        os.replace(tmp, last)
        print(f'   (resume với workers = {n})')


def run_yolo(model_key, seed, ctx, smoke=False):
    from ultralytics import YOLO
    import torch
    y = ctx['cfg']['yolo']
    base = ctx['results_dir'] / ('_smoke' if smoke else '') / model_key / f'seed{seed}'
    if (base / 'metrics.json').exists() and not smoke:
        print(f'⏭  {model_key} seed {seed} đã xong, bỏ qua ({base / "metrics.json"})')
        return
    if smoke and base.exists():
        import shutil
        shutil.rmtree(base)          # smoke test luôn chạy lại từ đầu
    base.mkdir(parents=True, exist_ok=True)
    save_json(ctx['env'], base / 'env.json')
    shutil_copy(CONFIG_PATH, base / 'config_used.yaml')
    data_yaml = resolved_data_yaml(ctx['ds'], base)
    seed_everything(seed)

    train_dir = base / 'train'
    last = train_dir / 'weights' / 'last.pt'
    done_flag = base / 'train_done.json'
    t0 = time.time()
    if not done_flag.exists():
        if last.exists() and not smoke:
            print(f'↻ Tiếp tục train {model_key} seed {seed} từ {last}')
            if ctx.get('workers_override') is not None:
                _set_resume_workers(last, ctx['workers_override'])
            YOLO(str(last)).train(resume=True)
        else:
            model = YOLO(YOLO_WEIGHTS[model_key])
            model.train(data=data_yaml, epochs=1 if smoke else y['epochs'], imgsz=y['imgsz'], batch=y['batch'],
                        workers=y['workers'], seed=seed, deterministic=y['deterministic'], amp=y['amp'],
                        optimizer=y['optimizer'], patience=y['patience'], fliplr=y['fliplr'], cache=y['cache'],
                        fraction=0.05 if smoke else 1.0, device=0, project=str(base), name='train',
                        exist_ok=True, plots=not smoke, verbose=True)
        save_json({'train_seconds_last_session': round(time.time() - t0, 1)}, done_flag)
    if smoke:
        print(f'✅ Smoke test {model_key}: train 1 epoch OK, VRAM tối đa {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
        return

    best = train_dir / 'weights' / 'best.pt'
    model = YOLO(str(best))
    # 1) Số liệu theo cách tính của Ultralytics (để đối chiếu)
    m = model.val(data=data_yaml, split='test', imgsz=y['imgsz'], batch=y['batch'], conf=y['eval_conf'],
                  iou=y['eval_iou'], max_det=y['max_det'], device=0, plots=True, project=str(base), name='val_test',
                  exist_ok=True)
    ultra = {k: float(v) for k, v in m.results_dict.items()}
    # 2) Số liệu COCO dùng chung cho mọi mô hình
    gt, path_to_id, items = build_coco_gt(ctx['ds']['root'], 'test', ctx['ds']['nc'])
    preds, t_inf, n_img = [], 0.0, 0
    # Dự đoán TỪNG ẢNH một (truyền cả danh sách thì Ultralytics gộp mọi ảnh vào 1 batch → hết VRAM)
    for p, _ in items:
        r = model.predict(source=str(p), imgsz=y['imgsz'], conf=y['eval_conf'], iou=y['eval_iou'],
                          max_det=y['max_det'], device=0, verbose=False)[0]
        iid = path_to_id[str(Path(p).resolve())]
        b = r.boxes
        for (x0, y0, x1, y1), c, s in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
            preds.append({'image_id': iid, 'category_id': int(c) + 1,
                          'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], 'score': float(s)})
        n_img += 1
    save_json(preds, base / 'test_predictions_coco.json')
    coco_res, per_class = coco_evaluate(gt, preds, ctx['ds']['names'])
    tr = json.load(open(done_flag, encoding='utf-8'))
    finish_run(base, ctx, model_key, seed, coco_res, per_class,
               {'ultralytics_test': ultra, 'ultra_mAP50_95': ultra.get('metrics/mAP50-95(B)'),
                'ultra_mAP50': ultra.get('metrics/mAP50(B)'), 'train_seconds_last_session': tr['train_seconds_last_session'],
                'n_test_images': n_img})


def shutil_copy(src, dst):
    import shutil
    shutil.copy2(src, dst)


# ----------------------------------------------------------------------------------
# Faster R-CNN ResNet50-FPN (torchvision)
# ----------------------------------------------------------------------------------
class YoloDetDataset:
    """Đọc ảnh bằng OpenCV (giống Ultralytics) và nhãn YOLO → box pixel xyxy, nhãn 1..nc (0 = nền)."""

    def __init__(self, root, split, limit=None):
        self.items = list_split(root, split)[:limit]
        self.n_dropped = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import torch
        ip, lp = self.items[i]
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Không đọc được ảnh {ip}')
        H, W = img.shape[:2]
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255)
        boxes, labels = [], []
        for c, xc, yc, w, h in read_yolo_label(lp):
            x0, y0 = max(0.0, (xc - w / 2) * W), max(0.0, (yc - h / 2) * H)
            x1, y1 = min(W, (xc + w / 2) * W), min(H, (yc + h / 2) * H)
            if x1 - x0 >= 1 and y1 - y0 >= 1:      # torchvision không nhận box suy biến
                boxes.append([x0, y0, x1, y1])
                labels.append(c + 1)
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  'labels': torch.tensor(labels, dtype=torch.int64), 'image_id': i}
        return img, target


def _collate(batch):
    return tuple(zip(*batch))


def _worker_init(worker_id):
    import torch
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def build_frcnn(nc, f):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, min_size=f['min_size'],
                                    max_size=f['max_size'], box_score_thresh=f['eval_score_thresh'],
                                    box_detections_per_img=f['max_det'])
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, nc + 1)
    return model


def frcnn_predict(model, root, split, nc, f, limit=None):
    import torch
    gt, path_to_id, items = build_coco_gt(root, split, nc)
    if limit:
        keep = {i + 1 for i in range(limit)}
        gt = {**gt, 'images': [im for im in gt['images'] if im['id'] in keep],
              'annotations': [a for a in gt['annotations'] if a['image_id'] in keep]}
    ds = YoloDetDataset(root, split, limit)
    dl = torch.utils.data.DataLoader(ds, batch_size=f['batch'], shuffle=False, num_workers=f['workers'],
                                     collate_fn=_collate)
    model.eval()
    preds = []
    with torch.inference_mode():
        for imgs, tgts in dl:
            imgs = [im.cuda(non_blocking=True) for im in imgs]
            with torch.autocast('cuda', enabled=f['amp']):
                outs = model(imgs)
            for t, o in zip(tgts, outs):
                iid = t['image_id'] + 1          # cùng thứ tự với build_coco_gt
                for (x0, y0, x1, y1), c, s in zip(o['boxes'].float().cpu().numpy(), o['labels'].cpu().numpy(),
                                                  o['scores'].float().cpu().numpy()):
                    preds.append({'image_id': iid, 'category_id': int(c),
                                  'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], 'score': float(s)})
    return gt, preds


def run_frcnn(seed, ctx, smoke=False):
    import torch
    f = ctx['cfg']['faster_rcnn']
    nc, root = ctx['ds']['nc'], ctx['ds']['root']
    base = ctx['results_dir'] / ('_smoke' if smoke else '') / FRCNN_KEY / f'seed{seed}'
    if (base / 'metrics.json').exists() and not smoke:
        print(f'⏭  {FRCNN_KEY} seed {seed} đã xong, bỏ qua')
        return
    if smoke and base.exists():
        import shutil
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    save_json(ctx['env'], base / 'env.json')
    shutil_copy(CONFIG_PATH, base / 'config_used.yaml')
    seed_everything(seed)

    model = build_frcnn(nc, f).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=f['lr'], momentum=f['momentum'], weight_decay=f['weight_decay'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=f['lr_steps'], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=f['amp'])
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = YoloDetDataset(root, 'train', 64 if smoke else None)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=f['batch'], shuffle=True, num_workers=f['workers'],
                                     collate_fn=_collate, generator=g, worker_init_fn=_worker_init,
                                     persistent_workers=f['workers'] > 0)
    epochs = 1 if smoke else f['epochs']
    last, best_path, log_path = base / 'last.pth', base / 'best.pth', base / 'train_log.csv'
    start_ep, best_map = 0, -1.0
    if last.exists() and not smoke:
        ck = torch.load(last, map_location='cuda', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['rng_cpu'].cpu())
        torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
        g.set_state(ck['rng_loader'].cpu())
        start_ep, best_map = ck['epoch'] + 1, ck['best_map']
        print(f'↻ Tiếp tục Faster R-CNN seed {seed} từ epoch {start_ep}')
    t0 = time.time()
    warm = min(f['warmup_iters'], len(dl) - 1)
    for ep in range(start_ep, epochs):
        model.train()
        tot, n = 0.0, 0
        for it, (imgs, tgts) in enumerate(dl):
            if ep == 0 and it < warm:
                for pg in opt.param_groups:
                    pg['lr'] = f['lr'] * (0.001 + (1 - 0.001) * it / warm)
            elif ep == 0 and it == warm:
                for pg in opt.param_groups:
                    pg['lr'] = f['lr']
            imgs = [im.cuda(non_blocking=True) for im in imgs]
            tgts = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in tgts]
            with torch.autocast('cuda', enabled=f['amp']):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            n += 1
            if it % 50 == 0:
                print(f'  epoch {ep + 1}/{epochs} iter {it}/{len(dl)} loss {float(loss):.4f}')
        sched.step()
        gt_v, pr_v = frcnn_predict(model, root, 'valid', nc, f, limit=64 if smoke else None)
        val_res, _ = coco_evaluate(gt_v, pr_v, ctx['ds']['names'])
        is_best = val_res['mAP50_95'] > best_map
        if is_best:
            best_map = val_res['mAP50_95']
            torch.save(model.state_dict(), best_path)
        with open(log_path, 'a', encoding='utf-8') as fh:
            if ep == 0:
                fh.write('epoch,train_loss,val_mAP50_95,val_mAP50,best\n')
            fh.write(f'{ep + 1},{tot / max(n, 1):.5f},{val_res["mAP50_95"]:.5f},{val_res["mAP50"]:.5f},{int(is_best)}\n')
        print(f'epoch {ep + 1}: loss {tot / max(n, 1):.4f} | valid mAP50-95 {val_res["mAP50_95"]:.4f} | mAP50 {val_res["mAP50"]:.4f}')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': ep, 'best_map': best_map,
                    'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(),
                    'rng_loader': g.get_state()}, last)
    if smoke:
        print(f'✅ Smoke test Faster R-CNN OK, VRAM tối đa {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
        return
    model.load_state_dict(torch.load(best_path, map_location='cuda'))
    gt, preds = frcnn_predict(model, root, 'test', nc, f)
    save_json(preds, base / 'test_predictions_coco.json')
    coco_res, per_class = coco_evaluate(gt, preds, ctx['ds']['names'])
    finish_run(base, ctx, FRCNN_KEY, seed, coco_res, per_class,
               {'best_val_mAP50_95': best_map, 'train_seconds_last_session': round(time.time() - t0, 1),
                'n_test_images': len(gt['images'])})


# ----------------------------------------------------------------------------------
# Điểm vào cho 3 file train
# ----------------------------------------------------------------------------------
def main(machine):
    ap = argparse.ArgumentParser(description=f'Train benchmark VN Traffic Sign – {machine["label"]}')
    ap.add_argument('--data', required=True, help='Thư mục đã giải nén split_data_by_box.zip hoặc split_data_by_image.zip')
    ap.add_argument('--check', action='store_true', help='Chỉ kiểm tra máy, cấu hình, dữ liệu rồi dừng')
    ap.add_argument('--smoke', action='store_true', help='Chạy thử 1 epoch trên ít ảnh để kiểm tra VRAM')
    ap.add_argument('--calib', action='store_true', help=f'Chạy hiệu chuẩn: {CALIB_MODEL} seed {CALIB_SEED} (giống nhau trên 3 máy)')
    ap.add_argument('--only', choices=machine['models'], help='Chỉ chạy một mô hình')
    ap.add_argument('--seed', type=int, help='Chỉ chạy một seed')
    ap.add_argument('--force-gpu', action='store_true', help='Cho chạy dù GPU không khớp máy dự kiến')
    ap.add_argument('--allow-version-mismatch', action='store_true')
    ap.add_argument('--workers', type=int, help='Ghi đè số DataLoader workers CHỈ trên máy này (máy thiếu RAM). '
                    'Không đổi config_sha256; được ghi vào env.json')
    args = ap.parse_args()

    cfg = load_config()
    vers = check_versions(args.allow_version_mismatch)
    env = env_info()
    print('=' * 70)
    print(f'Máy dự kiến : {machine["label"]}')
    print(f'GPU thực tế : {env["gpu_name"]} ({env["vram_gb"]} GB), driver {env["driver"]}, CUDA {env["cuda_runtime"]}')
    print(f'Thư viện    : ' + ', '.join(f'{k} {v}' for k, v in vers.items()))
    print(f'config_sha256 = {cfg["_sha256"]}   ← 3 máy phải giống nhau')
    if machine['gpu_hint'].lower() not in env['gpu_name'].lower():
        msg = f'GPU "{env["gpu_name"]}" không phải máy dự kiến ({machine["gpu_hint"]}) của file này.'
        if not args.force_gpu:
            raise SystemExit(f'❌ {msg} Chạy đúng file của máy mình, hoặc thêm --force-gpu nếu nhóm đã thống nhất đổi máy.')
        print(f'⚠️ {msg} (đang chạy vì --force-gpu)')
    ds = check_dataset(args.data, cfg, require_fingerprint=not (args.check or args.smoke))
    print(f'Dữ liệu     : {ds["root"]} ({ds["mode"] or "không rõ cách chia"}), {ds["nc"]} class')
    print(f'dataset_fingerprint = {ds["fingerprint"]}   ← 3 máy phải giống nhau')
    print('=' * 70)
    if args.check:
        for s in SPLITS:
            print(f'  {s}: {len(list_split(ds["root"], s))} ảnh')
        print('✅ Kiểm tra xong. Không train.')
        return

    if args.workers is not None:
        print(f'⚠️ Ghi đè workers = {args.workers} trên máy này (config ghi {cfg["yolo"]["workers"]}); đã lưu vào env.json')
        cfg['yolo']['workers'] = args.workers
        cfg['faster_rcnn']['workers'] = args.workers
        env['workers_override'] = args.workers
    ctx = {'cfg': cfg, 'env': env, 'ds': ds, 'machine': machine, 'workers_override': args.workers,
           'results_dir': (CODE_DIR / cfg['run']['results_dir']).resolve()}
    ctx['results_dir'].mkdir(parents=True, exist_ok=True)

    if args.calib:
        jobs = [(CALIB_MODEL, CALIB_SEED)]
        ctx['results_dir'] = ctx['results_dir'] / f'_calib_{machine["id"]}'
        ctx['results_dir'].mkdir(parents=True, exist_ok=True)
    else:
        models = [args.only] if args.only else machine['models']
        seeds = [args.seed] if args.seed is not None else cfg['run']['seeds']
        jobs = [(m, s) for m in models for s in ([seeds[0]] if args.smoke else seeds)]
    print('Danh sách chạy:', ', '.join(f'{m}/seed{s}' for m, s in jobs))
    for m, s in jobs:
        print(f'\n########## {m} | seed {s} ##########')
        if m == FRCNN_KEY:
            run_frcnn(s, ctx, smoke=args.smoke)
        else:
            run_yolo(m, s, ctx, smoke=args.smoke)


# ----------------------------------------------------------------------------------
# Tổng hợp kết quả từ nhiều máy:  python common.py aggregate <thư mục results> ...
# ----------------------------------------------------------------------------------
def aggregate(dirs):
    import pandas as pd
    rows = []
    for d in dirs:
        for p in Path(d).rglob('metrics.json'):
            if '_smoke' in p.parts:
                continue
            m = json.load(open(p, encoding='utf-8'))
            rows.append({'model': m['model'], 'seed': m['seed'], 'machine': m['machine'], 'gpu': m['gpu'],
                         'calib': any(part.startswith('_calib') for part in p.parts),
                         'config_sha256': m['config_sha256'], 'dataset_fingerprint': m['dataset_fingerprint'],
                         **{f'coco_{k}': v for k, v in m['coco_test'].items()}})
    if not rows:
        raise SystemExit('Không tìm thấy metrics.json nào.')
    df = pd.DataFrame(rows)
    out = Path('aggregate')
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'all_runs.csv', index=False, encoding='utf-8-sig')
    for col in ('config_sha256', 'dataset_fingerprint'):
        if df[col].nunique() > 1:
            print(f'⚠️ CẢNH BÁO: có {df[col].nunique()} giá trị {col} khác nhau → các lần chạy không cùng cấu hình/dữ liệu!')
    calib = df[df['calib']]
    if len(calib):
        print('\nHiệu chuẩn (cùng mô hình, cùng seed, khác máy):')
        print(calib[['machine', 'gpu', 'coco_mAP50_95', 'coco_mAP50']].to_string(index=False))
        calib.to_csv(out / 'calibration.csv', index=False, encoding='utf-8-sig')
    main_runs = df[~df['calib']]
    metric_cols = [c for c in main_runs.columns if c.startswith('coco_')]
    agg = main_runs.groupby('model')[metric_cols].agg(['mean', 'std', 'count'])
    agg.to_csv(out / 'mean_std_by_model.csv', encoding='utf-8-sig')
    pretty = pd.DataFrame({c: main_runs.groupby('model')[c].agg(lambda s: f'{s.mean():.4f} ± {s.std(ddof=1):.4f}' if len(s) > 1 else f'{s.mean():.4f}')
                           for c in metric_cols})
    pretty['seeds'] = main_runs.groupby('model')['seed'].count()
    pretty['machines'] = main_runs.groupby('model')['machine'].agg(lambda s: ','.join(sorted(set(s))))
    pretty.to_csv(out / 'mean_std_pretty.csv', encoding='utf-8-sig')
    print('\nKết quả trên test (mean ± std qua các seed):')
    print(pretty.to_string())
    print(f'\nĐã lưu vào {out.resolve()}')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'aggregate':
        aggregate(sys.argv[2:])
    else:
        print('Dùng: python common.py aggregate <thư mục results 1> <thư mục results 2> ...')
