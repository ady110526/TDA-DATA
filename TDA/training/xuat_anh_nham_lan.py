"""
XUẤT ẢNH CÁC LỖI CỦA MÔ HÌNH ĐỂ NHÓM XEM BẰNG MẮT (YOLO26s, một seed).

Tạo 2 phần:
  (1) nham_lop/ : MỌI trường hợp hộp thật bị nhận NHẦM LỚP (lớp A thành lớp B).
  (2) o_tu_10/  : các ô của ma trận nhầm lẫn có từ --nguong trường hợp trở lên (mặc định 10),
                  gồm cả ô "bỏ sót" (cột nền) và ô "báo nhầm" (hàng nền). Mỗi ô được đánh số O01, O02, ...
                  theo số lượng giảm dần, và được in ra màn hình.

Chạy trên máy có dữ liệu và kết quả của YOLO26s (máy 1), trong thư mục code (cạnh common.py):
    python xuat_anh_nham_lan.py --data <thư mục split_data_by_box> ^
        --pred results\\yolo26s\\seed1\\test_predictions_coco.json --conf 0.37

--conf: ngưỡng tin cậy đã chọn trên tập valid cho đúng lần chạy đó (YOLO26s seed 1: 0.37).
Cách ghép giống hệt ma trận nhầm lẫn trong bài: dự đoán có điểm >= conf, IoU >= 0.5,
không phân biệt lớp, ghép theo điểm tin cậy giảm dần, mỗi hộp thật chỉ ghép một lần.

Trong ảnh cắt: khung XANH LÁ = hộp thật (nhãn), khung ĐỎ = dự đoán của mô hình.

Kết quả (thư mục kiem_tra_nham_lan/):
  tong_quan.txt               – các con số kiểm tra + danh sách ô >= ngưỡng (giống phần in ra màn hình)
  ma_tran_nham_lan.csv        – ma trận 116 x 116 (115 lớp + nền) để đối chiếu với hình trong bài
  danh_sach_nham_lop.csv      – phần (1), mỗi dòng một trường hợp
  danh_sach_o_tu_10.csv       – phần (2), mỗi dòng một trường hợp, có cột "chan_doan_tu_dong"
  Nhóm điền cột "ket_luan" (một trong các giá trị dưới) và "ghi_chu" nếu cần:
        nhan_sai    : hộp thật bị gán sai lớp / sai vị trí / không phải biển
        thieu_nhan  : (ô báo nhầm) chỗ đó có biển thật nhưng chưa được gán nhãn
        qua_nho_mo  : biển quá nhỏ, mờ, bị che, ngược sáng – mắt người cũng khó nhận
        mo_hinh_sai : nhãn đúng, biển nhìn rõ, nhưng mô hình sai
        khong_ro    : không phân định được
Không train, không cần GPU.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0][:, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1][:, None], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def short_name(n):
    """Mã ngắn của lớp, giống notebook EDA: 127-toc-do-...-60 → 127-60; R.E-11a-... → R.E-11a; 103c-cam-... → 103c."""
    if n.startswith('R.E-11a'):
        return 'R.E-11a'
    if n.startswith(('127-', '306-')):
        return n.split('-')[0] + '-' + n.split('-')[-1]
    return n.split('-')[0]


def match_all(gt, preds, conf, nc, iou_thr=0.5):
    """Ghép giống ma trận trong bài. Trả về ma trận (nc+1)x(nc+1) và danh sách sự kiện:
    ('nham_lop', iid, g, gbox, d, dbox, score, iou, chẩn đoán)
    ('bo_sot',   iid, g, gbox, d_hint, dbox_hint, score_hint, iou_hint, chẩn đoán)
    ('bao_nham', iid, g_hint, gbox_hint, d, dbox, score, iou_hint, chẩn đoán)"""
    by_gt, by_all = {}, {}
    for a in gt['annotations']:
        by_gt.setdefault(a['image_id'], []).append((a['category_id'] - 1, a['bbox']))
    for p in preds:
        by_all.setdefault(p['image_id'], []).append((p['category_id'] - 1, p['bbox'], p['score']))
    cm = np.zeros((nc + 1, nc + 1), int)
    ev = []
    for im in gt['images']:
        iid = im['id']
        G = by_gt.get(iid, [])
        A = sorted(by_all.get(iid, []), key=lambda x: -x[2])       # mọi dự đoán (kể cả dưới ngưỡng)
        D = [x for x in A if x[2] >= conf]                          # dự đoán được tính
        gc = np.array([c for c, _ in G], int)
        gb = np.array([b for _, b in G], float).reshape(-1, 4)
        db = np.array([b for _, b, _ in D], float).reshape(-1, 4)
        M = iou_matrix(db, gb)
        used = np.zeros(len(G), bool)
        matched_dt = np.zeros(len(D), bool)
        for i, (dc, dbox, s) in enumerate(D):
            ok = (~used) & (M[i] >= iou_thr) if len(G) else np.zeros(0, bool)
            if ok.any():
                j = np.flatnonzero(ok)[np.argmax(M[i, ok])]
                used[j] = True
                matched_dt[i] = True
                cm[gc[j], dc] += 1
                if gc[j] != dc:
                    ev.append(('nham_lop', iid, int(gc[j]), list(gb[j]), int(dc), list(dbox), float(s), float(M[i, j]), ''))
        # báo nhầm: dự đoán >= conf không ghép được hộp thật nào
        for i, (dc, dbox, s) in enumerate(D):
            if matched_dt[i]:
                continue
            cm[nc, dc] += 1
            if len(G):
                j = int(np.argmax(M[i]))
                mi = float(M[i, j])
            else:
                j, mi = -1, 0.0
            if mi >= iou_thr:
                why = 'trung_lap: hộp thật ở đây đã được một dự đoán khác (điểm cao hơn) ghép'
            elif mi >= 0.1:
                why = f'lech_hop: chồng lên hộp thật {mi:.2f} IoU (< 0.5) – vị trí/kích thước hộp lệch'
            else:
                why = 'khong_co_hop_that_gan: không có nhãn nào gần – có thể THIẾU NHÃN hoặc mô hình báo nhầm vật khác'
            ev.append(('bao_nham', iid, int(gc[j]) if j >= 0 and mi > 0 else -1, list(gb[j]) if j >= 0 and mi > 0 else None,
                       int(dc), list(dbox), float(s), mi, why))
        # bỏ sót: hộp thật không được ghép
        ab = np.array([b for _, b, _ in A], float).reshape(-1, 4)
        MA = iou_matrix(ab, gb)
        for j in np.flatnonzero(~used):
            cm[gc[j], nc] += 1
            hint = None
            if len(A):
                col = MA[:, j]
                low = [k for k in range(len(A)) if col[k] >= iou_thr and A[k][2] < conf]
                hi_taken = [k for k in range(len(A)) if col[k] >= iou_thr and A[k][2] >= conf]
                near = [k for k in range(len(A)) if 0.1 <= col[k] < iou_thr and A[k][2] >= conf]
                if low:
                    k = max(low, key=lambda k: A[k][2])
                    why = f'diem_thap: mô hình CÓ thấy (IoU {col[k]:.2f}) nhưng điểm {A[k][2]:.2f} < ngưỡng {conf}'
                    hint = k
                elif hi_taken:
                    k = hi_taken[0]
                    why = 'bi_chiem: dự đoán trùng chỗ đã được ghép cho hộp thật khác bên cạnh (hai biển sát nhau)'
                    hint = k
                elif near:
                    k = max(near, key=lambda k: col[k])
                    why = f'lech_hop: có dự đoán nhưng IoU chỉ {col[k]:.2f} (< 0.5)'
                    hint = k
                else:
                    why = 'khong_phat_hien: không có dự đoán nào trùng chỗ, kể cả điểm rất thấp'
            else:
                why = 'khong_phat_hien: ảnh không có dự đoán nào'
            if hint is None:
                ev.append(('bo_sot', iid, int(gc[j]), list(gb[j]), -1, None, 0.0, 0.0, why))
            else:
                ev.append(('bo_sot', iid, int(gc[j]), list(gb[j]), int(A[hint][0]), list(A[hint][1]), float(A[hint][2]),
                           float(MA[hint, j]), why))
    return cm, ev


def crop_pair(img_path, gbox, dbox, out_path, size=320):
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        W, H = im.size
        boxes = [b for b in (gbox, dbox) if b is not None]
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        side = max(x1 - x0, y1 - y0) * 2.5 + 20          # lấy thêm bối cảnh xung quanh
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = [max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2)]
        crop = im.crop(tuple(int(v) for v in box))
        d = ImageDraw.Draw(crop)
        for b, col in ((gbox, (0, 200, 0)), (dbox, (230, 0, 0))):
            if b is not None:
                d.rectangle([b[0] - box[0], b[1] - box[1], b[0] + b[2] - box[0], b[1] + b[3] - box[1]], outline=col, width=2)
        crop = ImageOps.pad(crop, (size, size), color=(245, 245, 245))
        crop.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='thư mục split_data_by_box (cần để cắt ảnh)')
    ap.add_argument('--pred', required=True)
    ap.add_argument('--conf', type=float, required=True)
    ap.add_argument('--nguong', type=int, default=10, help='ô có từ bấy nhiêu trường hợp trở lên thì xuất ảnh (mặc định 10)')
    ap.add_argument('--out', default='kiem_tra_nham_lan')
    ap.add_argument('--gt', help='(chỉ để kiểm thử) tệp test_gt_coco.json thay cho --data; khi đó không cắt ảnh')
    ap.add_argument('--names', help='(chỉ để kiểm thử) tệp data.yaml để lấy tên lớp khi dùng --gt')
    args = ap.parse_args()

    if args.gt:
        import yaml
        gt = json.load(open(args.gt, encoding='utf-8'))
        y = yaml.safe_load(open(args.names, encoding='utf-8'))
        names = list(y['names'].values()) if isinstance(y['names'], dict) else list(y['names'])
        id2path = None
    else:
        import common
        cfg = common.load_config()
        ds = common.check_dataset(args.data, cfg, require_fingerprint=True)
        gt, _, items = common.build_coco_gt(ds['root'], 'test', ds['nc'])
        names = ds['names']
        id2path = {i + 1: p for i, (p, _) in enumerate(items)}
    nc = len(names)
    short = [short_name(n) for n in names] + ['nen']
    preds = json.load(open(args.pred, encoding='utf-8'))
    cm, ev = match_all(gt, preds, args.conf, nc)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fname = {im['id']: (Path(id2path[im['id']]).name if id2path else im['file_name']) for im in gt['images']}

    # ---- con số kiểm tra (phải khớp với ma trận trong bài) ----
    n_gt = len(gt['annotations'])
    n_conf = int(cm[:nc, :nc].sum() - np.trace(cm[:nc, :nc]))
    lines = [f'Tệp dự đoán : {args.pred}', f'Ngưỡng tin cậy: {args.conf}   |   ngưỡng ô: >= {args.nguong}',
             f'Số hộp thật tập test : {n_gt}',
             f'  đúng lớp           : {int(np.trace(cm[:nc, :nc]))}',
             f'  nhầm lớp           : {n_conf}',
             f'  bỏ sót (cột nền)   : {int(cm[:nc, nc].sum())}',
             f'Báo nhầm (hàng nền)  : {int(cm[nc, :nc].sum())}', '']

    # ---- các ô >= ngưỡng, đánh số O01, O02, ... ----
    cells = [(int(cm[i, j]), i, j) for i in range(nc + 1) for j in range(nc + 1)
             if i != j and cm[i, j] >= args.nguong]
    cells.sort(key=lambda x: (-x[0], x[1], x[2]))
    cell_id = {}
    lines.append(f'CÁC Ô CÓ TỪ {args.nguong} TRƯỜNG HỢP TRỞ LÊN: {len(cells)} ô')
    lines.append(f'{"số":<5}{"loại":<10}{"lớp thật":<12}{"dự đoán":<12}{"số ca":>6}{"tổng hộp thật của lớp":>24}')
    for k, (n, i, j) in enumerate(cells, 1):
        loai = 'bỏ sót' if j == nc else ('báo nhầm' if i == nc else 'nhầm lớp')
        tong = int(cm[i].sum()) if i < nc else '-'
        cid = f'O{k:02d}'
        cell_id[(i, j)] = cid
        lines.append(f'{cid:<5}{loai:<10}{short[i]:<12}{short[j]:<12}{n:>6}{str(tong):>24}')
    txt = '\n'.join(lines)
    print(txt)
    (out / 'tong_quan.txt').write_text(txt + '\n', encoding='utf-8')

    with open(out / 'ma_tran_nham_lan.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['that \\ du_doan'] + short)
        for i in range(nc + 1):
            w.writerow([short[i]] + [int(v) for v in cm[i]])

    def lop(i):
        return short[i] if i is not None and i >= 0 else ''

    def save_rows(path, rows):
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['stt'])
            w.writeheader()
            w.writerows(rows)

    # ---- (1) nhầm lớp: mọi trường hợp ----
    rows1, counts = [], {}
    nham = sorted([e for e in ev if e[0] == 'nham_lop'], key=lambda e: (e[2], e[4]))
    for k, (_, iid, g, gbox, d, dbox, s, iou, _) in enumerate(nham, 1):
        pair = f'{short[g]}__thanh__{short[d]}'
        counts[pair] = counts.get(pair, 0) + 1
        fn = f'{counts[pair]:02d}_{Path(fname[iid]).stem[:60]}.jpg'
        if id2path:
            (out / 'nham_lop' / pair).mkdir(parents=True, exist_ok=True)
            crop_pair(id2path[iid], gbox, dbox, out / 'nham_lop' / pair / fn)
        rows1.append({'stt': k, 'cap': pair, 'lop_that': short[g], 'mo_hinh_doan': short[d], 'diem_tin_cay': round(s, 3),
                      'iou': round(iou, 3), 'anh_goc': fname[iid], 'file_cat': f'nham_lop/{pair}/{fn}',
                      'ket_luan': '', 'ghi_chu': ''})
    save_rows(out / 'danh_sach_nham_lop.csv', rows1)

    # ---- (2) các ô >= ngưỡng ----
    rows2, cnt2 = [], {}
    for e in ev:
        typ, iid, g, gbox, d, dbox, s, iou, why = e
        if typ == 'nham_lop':
            key = (g, d)
        elif typ == 'bo_sot':
            key = (g, nc)
        else:
            key = (nc, d)
        if key not in cell_id:
            continue
        cid = cell_id[key]
        loai = {'nham_lop': 'nham_lop', 'bo_sot': 'bo_sot', 'bao_nham': 'bao_nham'}[typ]
        folder = f'{cid}_{loai}_{short[key[0]]}__{short[key[1]]}'
        cnt2[folder] = cnt2.get(folder, 0) + 1
        fn = f'{cnt2[folder]:02d}_{Path(fname[iid]).stem[:60]}.jpg'
        if id2path:
            (out / 'o_tu_10' / folder).mkdir(parents=True, exist_ok=True)
            crop_pair(id2path[iid], gbox, dbox, out / 'o_tu_10' / folder / fn)
        rows2.append({'o': cid, 'loai': loai, 'lop_that': lop(g) if typ != 'bao_nham' else '',
                      'mo_hinh_doan': lop(d) if typ != 'bo_sot' else '',
                      'du_doan_gan_nhat': (f'{lop(d)} ({s:.2f})' if typ == 'bo_sot' and d >= 0 else
                                           (f'hộp thật gần nhất: {lop(g)}' if typ == 'bao_nham' and g >= 0 else '')),
                      'diem_tin_cay': round(s, 3), 'iou': round(iou, 3),
                      'chan_doan_tu_dong': why, 'anh_goc': fname[iid], 'file_cat': f'o_tu_10/{folder}/{fn}',
                      'ket_luan': '', 'ghi_chu': ''})
    rows2.sort(key=lambda r: (r['o'], r['file_cat']))
    save_rows(out / 'danh_sach_o_tu_10.csv', rows2)

    # tóm tắt chẩn đoán tự động theo ô
    summ = {}
    for r in rows2:
        summ.setdefault(r['o'], {}).setdefault(r['chan_doan_tu_dong'].split(':')[0], 0)
        summ[r['o']][r['chan_doan_tu_dong'].split(':')[0]] += 1
    extra = ['', 'CHẨN ĐOÁN TỰ ĐỘNG THEO Ô (chỉ là gợi ý, nhóm vẫn phải xem ảnh):']
    for cid in sorted(summ):
        extra.append(f'  {cid}: ' + ', '.join(f'{k} {v}' for k, v in sorted(summ[cid].items(), key=lambda x: -x[1])))
    extra.append('')
    extra.append(f'✅ {len(rows1)} trường hợp nhầm lớp → danh_sach_nham_lop.csv' + ('' if id2path else ' (không cắt ảnh)'))
    extra.append(f'✅ {len(rows2)} trường hợp trong {len(cells)} ô >= {args.nguong} → danh_sach_o_tu_10.csv')
    extra.append(f'   Thư mục kết quả: {out.resolve()}')
    print('\n'.join(extra))
    with open(out / 'tong_quan.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(extra) + '\n')


if __name__ == '__main__':
    main()
