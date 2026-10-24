"""Hàm dùng chung: đọc nhãn thật/dự đoán định dạng COCO, ghép dự đoán với hộp thật,
tính Precision/Recall/F1 (vi mô, vĩ mô) và ma trận nhầm lẫn ở một ngưỡng tin cậy."""
import json

import numpy as np

IOU_THR = 0.5


def iou_matrix(a, b):
    """a: N×4, b: M×4 theo dạng [x, y, w, h] → ma trận IoU N×M."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0][:, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1][:, None], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def load_gt(path):
    """Trả về (danh sách image_id, {image_id: (lớp[], hộp[])}, số lớp, tên lớp hoặc None)."""
    g = json.load(open(path, encoding='utf-8'))
    by = {}
    for a in g['annotations']:
        c, b = by.setdefault(a['image_id'], ([], []))
        c.append(a['category_id'] - 1)
        b.append(a['bbox'])
    by = {k: (np.array(c, int), np.array(b, float).reshape(-1, 4)) for k, (c, b) in by.items()}
    return [im['id'] for im in g['images']], by, len(g['categories']), g.get('class_names')


def load_preds(path):
    """Gom dự đoán theo ảnh, sắp theo điểm giảm dần: {image_id: (lớp[], hộp[], điểm[])}."""
    by = {}
    for p in json.load(open(path, encoding='utf-8')):
        by.setdefault(p['image_id'], []).append((p['category_id'] - 1, p['bbox'], p['score']))
    out = {}
    for k, v in by.items():
        v.sort(key=lambda x: -x[2])
        out[k] = (np.array([x[0] for x in v], int), np.array([x[1] for x in v], float).reshape(-1, 4),
                  np.array([x[2] for x in v], float))
    return out


def evaluate(preds, img_ids, gt, nc, conf, iou_thr=IOU_THR):
    """Ở ngưỡng conf: (tp, fp, fn) theo lớp và ma trận nhầm lẫn (nc+1)×(nc+1).
    F1: ghép tham lam theo điểm, chỉ với hộp thật CÙNG lớp, IoU ≥ iou_thr, mỗi hộp thật ghép một lần.
    Ma trận nhầm lẫn: ghép KHÔNG phân biệt lớp; hàng = lớp thật, cột = lớp dự đoán, chỉ số nc = nền
    (hàng nền: báo nhầm; cột nền: bỏ sót)."""
    tp, fp, fn = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    empty = (np.zeros(0, int), np.zeros((0, 4)))
    for iid in img_ids:
        gc, gb = gt.get(iid, empty)
        dc, db, ds = preds.get(iid, (np.zeros(0, int), np.zeros((0, 4)), np.zeros(0)))
        keep = ds >= conf
        dc, db = dc[keep], db[keep]
        M = iou_matrix(db, gb)
        used = np.zeros(len(gc), bool)
        used2 = np.zeros(len(gc), bool)
        for i in range(len(dc)):
            ok = (~used) & (gc == dc[i]) & (M[i] >= iou_thr)
            if ok.any():
                j = np.flatnonzero(ok)[np.argmax(M[i, ok])]
                used[j] = True
                tp[dc[i]] += 1
            else:
                fp[dc[i]] += 1
            ok2 = (~used2) & (M[i] >= iou_thr)
            if ok2.any():
                j = np.flatnonzero(ok2)[np.argmax(M[i, ok2])]
                used2[j] = True
                cm[gc[j], dc[i]] += 1
            else:
                cm[nc, dc[i]] += 1
        for j in np.flatnonzero(~used):
            fn[gc[j]] += 1
        for j in np.flatnonzero(~used2):
            cm[gc[j], nc] += 1
    return tp, fp, fn, cm


def summarize(tp, fp, fn):
    """Precision/Recall/F1 vi mô và vĩ mô (vĩ mô: trung bình trên các lớp có hộp thật)."""
    P = tp.sum() / max(tp.sum() + fp.sum(), 1)
    R = tp.sum() / max(tp.sum() + fn.sum(), 1)
    F = 2 * P * R / max(P + R, 1e-12)
    has = (tp + fn) > 0
    pc = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rc = tp / np.maximum(tp + fn, 1)
    fc = np.where(pc + rc > 0, 2 * pc * rc / np.maximum(pc + rc, 1e-12), 0.0)
    return {'P_micro': P, 'R_micro': R, 'F1_micro': F, 'P_macro': pc[has].mean(),
            'R_macro': rc[has].mean(), 'F1_macro': fc[has].mean()}, fc
