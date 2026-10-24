# %% [markdown]
# # 02 – EDA cho bộ dữ liệu VN Traffic Sign (cleaned_data.zip)
#
# **Đầu vào:** `MyDrive/VNTrafficSign/output_cleaneddata/cleaned_data.zip` (kết quả của notebook `01_clean_data`).
#
# **Đầu ra:** `MyDrive/VNTrafficSign/output_eda/`
# - `figures/`: toàn bộ biểu đồ (PNG, 200 dpi)
# - `tables/`: toàn bộ bảng số liệu (CSV), để mọi con số trong bài báo đều sinh ra từ notebook này
# - `samples_per_class/`: lưới ảnh mẫu cho từng class (có vẽ bounding box)
# - `crops_per_class/`: ảnh crop từng biển theo class, dùng để soát nhầm nhãn bằng mắt
# - `summary.json`: các con số tổng hợp chính
#
# **Nguyên tắc:**
# - Gộp train + valid + test thành **một** tập ngay ở bước 0. Sau bước gộp, notebook không còn giữ bất kỳ thông tin nào về train/valid/test; toàn bộ EDA chạy trên tập gộp.
# - Mục ảnh gần trùng lặp (hashing) tạm thời chưa làm, theo yêu cầu của nhóm.
# - Giải nén vào `/content` (ổ cục bộ của Colab), không giải nén lên Drive.
# - Phân nhóm biển theo **QCVN 41:2024/BGTVT** và các quyết định nhóm đã chốt (xem mục 2).
# - Không có metadata nguồn ảnh, điều kiện chụp hay người gán nhãn (định dạng YOLO không chứa các thông tin này), nên notebook không phân tích các mục đó.
#
# | Mục | Nội dung |
# |---|---|
# | 0 | Giải nén, gộp train/valid/test thành 1 |
# | 1 | Bảng phân nhóm biển báo |
# | 2 | Đọc nhãn và kiểm tra toàn vẹn |
# | 3 | Thống kê tổng quan |
# | 4 | Phân bố class (thang log, head/medium/tail) |
# | 5 | Phân bố theo 5 nhóm biển báo |
# | 6 | Số box mỗi ảnh |
# | 7 | Kích thước ảnh |
# | 8 | Kích thước box (pixel, diện tích tương đối, COCO small/medium/large) |
# | 9 | Tỉ lệ khung box (w/h) – chỉ vẽ phân bố |
# | 10 | Heatmap vị trí tâm box |
# | 11 | Ma trận đồng xuất hiện |
# | 12 | Ảnh mẫu mỗi class + ảnh crop |
# | 13 | Box bất thường theo class (kích thước và tỉ lệ) |
# | 14 | Xuất bảng tổng hợp |

# %% [markdown]
# ## Cấu hình

# %%
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageOps

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive/VNTrafficSign')
    LOCAL_ROOT = Path('/content')
except ImportError:  # chạy ngoài Colab (dùng để kiểm thử)
    DRIVE_ROOT = Path(os.environ.get('VNTS_DRIVE_ROOT', './VNTrafficSign'))
    LOCAL_ROOT = Path(os.environ.get('VNTS_LOCAL_ROOT', './local'))

CLEAN_ZIP = DRIVE_ROOT / 'output_cleaneddata' / 'cleaned_data.zip'
EDA_OUT = DRIVE_ROOT / 'output_eda'
EXTRACT_DIR = LOCAL_ROOT / 'cleaned_data'     # giải nén cục bộ
MERGED_DIR = LOCAL_ROOT / 'merged_data'       # tập gộp train+valid+test (cục bộ)

FIG_DIR, TAB_DIR = EDA_OUT / 'figures', EDA_OUT / 'tables'
SAMPLE_DIR, CROP_DIR = EDA_OUT / 'samples_per_class', EDA_OUT / 'crops_per_class'
for d in (FIG_DIR, TAB_DIR, SAMPLE_DIR, CROP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Tham số phân tích (có thể chỉnh) ----
SEED = 42
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
HEAD_MIN, TAIL_MAX = 100, 20          # head: >=100 box | medium: 20–99 | tail: <20
COCO_SMALL, COCO_MEDIUM = 32 ** 2, 96 ** 2
TINY_BOX_PX = 2                       # box có cạnh < 2 px coi là kích thước ≈ 0
DUP_BOX_IOU = 0.9                     # 2 box cùng class trong 1 ảnh có IoU >= 0.9 coi là trùng lặp
N_SAMPLES_PER_CLASS = 4               # số ảnh mẫu mỗi class
N_CROPS_PER_CLASS = 32                # số crop mỗi class
ANOM_MIN_BOXES = 10                   # chỉ tìm box bất thường ở class có >= 10 box
ANOM_Z = 3.5                          # ngưỡng robust z-score (median/MAD)

# ---- Chọn ảnh đại diện cho Hình tổng quan 115 lớp (mục 12c) ----
MONTAGE_N_CANDIDATES = 24             # số ứng viên mỗi lớp (các box lớn nhất của lớp), đánh số #1, #2, ...
MONTAGE_CHOICE_CSV = EDA_OUT / 'montage_chon_anh.csv'   # file nhóm điền (xem hướng dẫn ở mục 12c); chưa có thì dùng #1
CAND_DIR = EDA_OUT / 'montage_ung_vien'                 # trang ứng viên của từng lớp

random.seed(SEED)
np.random.seed(SEED)

# Màu 5 nhóm biển (thứ tự cố định)
GROUP_ORDER = ['Cấm', 'Nguy hiểm', 'Hiệu lệnh', 'Chỉ dẫn', 'Phụ']
GROUP_COLOR = dict(zip(GROUP_ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
SEQ_CMAP = 'Blues'

plt.rcParams.update({
    'figure.dpi': 100, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
})


def savefig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.png')
    plt.show()


def savetab(df, name, index=False):
    df.to_csv(TAB_DIR / f'{name}.csv', index=index, encoding='utf-8-sig')


assert CLEAN_ZIP.exists(), f'Không tìm thấy {CLEAN_ZIP}. Hãy chạy notebook 01_clean_data trước.'
print('Đầu vào :', CLEAN_ZIP)
print('Đầu ra  :', EDA_OUT)

# %% [markdown]
# ## 0. Giải nén và gộp train + valid + test thành một tập
#
# Ảnh và nhãn của 3 split được chuyển vào chung `/content/merged_data/images` và `/content/merged_data/labels`.
# Nếu có tên file trùng giữa các split, notebook **dừng lại và báo** để nhóm quyết định (không tự đổi tên).
# Sau khi gộp, thư mục train/valid/test được xóa và không lưu lại thông tin ảnh thuộc split nào. Notebook kiểm tra lại điều này ở cuối ô.

# %%
for d in (EXTRACT_DIR, MERGED_DIR):
    if d.exists():
        shutil.rmtree(d)
EXTRACT_DIR.mkdir(parents=True)
with zipfile.ZipFile(CLEAN_ZIP) as zf:
    zf.extractall(EXTRACT_DIR)

yaml_files = list(EXTRACT_DIR.rglob('data.yaml'))
assert len(yaml_files) == 1, f'Cần đúng 1 data.yaml, tìm thấy {len(yaml_files)}'
DATA_ROOT = yaml_files[0].parent
data_cfg = yaml.safe_load(open(yaml_files[0], encoding='utf-8'))
CLASS_NAMES = list(data_cfg['names'].values()) if isinstance(data_cfg['names'], dict) else list(data_cfg['names'])
NC = int(data_cfg['nc'])
assert NC == len(CLASS_NAMES)
SPLITS = sorted(p.parent.name for p in DATA_ROOT.glob('*/images') if p.is_dir())
print('nc =', NC, '| split gốc:', SPLITS)

# Kiểm tra trùng tên file giữa các split trước khi gộp
stem_to_split = defaultdict(list)
for s in SPLITS:
    for p in (DATA_ROOT / s / 'images').iterdir():
        if p.suffix.lower() in IMG_EXTS:
            stem_to_split[p.stem].append(s)
clash = {k: v for k, v in stem_to_split.items() if len(v) > 1}
if clash:
    savetab(pd.DataFrame([(k, ','.join(v)) for k, v in clash.items()], columns=['stem', 'splits']), 'merge_name_clashes')
    raise RuntimeError(f'Có {len(clash)} tên file trùng giữa các split (xem tables/merge_name_clashes.csv). '
                       'Cần nhóm quyết định cách xử lý trước khi gộp.')

(MERGED_DIR / 'images').mkdir(parents=True)
(MERGED_DIR / 'labels').mkdir(parents=True)
merge_rows = []
for s in SPLITS:
    for p in sorted((DATA_ROOT / s / 'images').iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        shutil.move(str(p), MERGED_DIR / 'images' / p.name)
        lp = DATA_ROOT / s / 'labels' / f'{p.stem}.txt'
        has_lbl = lp.exists()
        if has_lbl:
            shutil.move(str(lp), MERGED_DIR / 'labels' / lp.name)
        merge_rows.append({'image': p.name, 'stem': p.stem, 'has_label_file': has_lbl})

images = pd.DataFrame(merge_rows)

# Xóa các thư mục split cũ (đã rỗng ảnh/nhãn) để không còn cấu trúc train/valid/test
for s in SPLITS:
    shutil.rmtree(DATA_ROOT / s)

# Kiểm tra: tập gộp là một thư mục phẳng, không còn split nào
n_img_merged = sum(1 for p in (MERGED_DIR / 'images').iterdir() if p.suffix.lower() in IMG_EXTS)
assert n_img_merged == len(images), 'Số ảnh trong tập gộp không khớp'
assert not any((DATA_ROOT / s).exists() for s in SPLITS), 'Vẫn còn thư mục split cũ'
assert not any(p.is_dir() for p in (MERGED_DIR / 'images').iterdir()), 'Tập gộp còn thư mục con'
assert 'orig_split' not in images.columns
print(f'✅ Đã gộp {len(images)} ảnh thành một tập duy nhất ở {MERGED_DIR} (không còn train/valid/test)')

# %% [markdown]
# ## 1. Bảng phân nhóm biển báo (QCVN 41:2024)
#
# Quy tắc đã chốt với nhóm (15/10/2026), áp dụng cho 115 class của bản v4:
# - Theo chữ số đầu của mã: **1xx cấm, 2xx nguy hiểm, 3xx hiệu lệnh, 4xx chỉ dẫn, 5xx phụ**.
# - Ngoại lệ thuộc **hiệu lệnh**: `122`, `403e`, `412f`, `420`, `421`, `R.E-11a`.
# - `133`, `134`, `135` (mã DP, hết cấm), `124e`, `131c` → cấm; `437`, `438`, `445g` → chỉ dẫn; `215b`, `221b`, `228b` → nguy hiểm (đều đúng theo chữ số đầu).
# - Bản v4 đã bỏ `468c` và `508` theo ý của nhóm.
#
# Kết quả mong đợi: **47 cấm, 33 nguy hiểm, 19 hiệu lệnh, 15 chỉ dẫn, 1 phụ** (tổng 115). Notebook sẽ dừng nếu có class không phân được nhóm.

# %%
HIEU_LENH_EXCEPTIONS = {'122', '403e', '412f', '420', '421', 'R.E-11a'}
DIGIT_TO_GROUP = {'1': 'Cấm', '2': 'Nguy hiểm', '3': 'Hiệu lệnh', '4': 'Chỉ dẫn', '5': 'Phụ'}
EXPECTED_GROUP_COUNTS = {'Cấm': 47, 'Nguy hiểm': 33, 'Hiệu lệnh': 19, 'Chỉ dẫn': 15, 'Phụ': 1}


def sign_code(name):
    return 'R.E-11a' if name.startswith('R.E-11a') else name.split('-')[0]


def sign_group(name):
    code = sign_code(name)
    if code in HIEU_LENH_EXCEPTIONS:
        return 'Hiệu lệnh'
    return DIGIT_TO_GROUP.get(code[0], None)


classes = pd.DataFrame({'class_id': range(NC), 'class_name': CLASS_NAMES})
classes['code'] = classes['class_name'].map(sign_code)
classes['group'] = classes['class_name'].map(sign_group)
# Nhãn ngắn dùng trên hình: mã biển, thêm hậu tố khi nhiều class cùng mã (vd. 127-50, 306-40)
_dup_code = classes['code'].duplicated(keep=False)
classes['short'] = np.where(_dup_code, classes['code'] + '-' + classes['class_name'].str.split('-').str[-1], classes['code'])
unmapped = classes[classes['group'].isna()]
assert unmapped.empty, f'Có class chưa phân được nhóm, cần nhóm xác nhận:\n{unmapped}'

group_n_classes = classes['group'].value_counts().reindex(GROUP_ORDER, fill_value=0)
display(group_n_classes.rename('số class').to_frame())
if group_n_classes.to_dict() != EXPECTED_GROUP_COUNTS:
    print('⚠️ CẢNH BÁO: số class theo nhóm khác kết quả đã chốt', EXPECTED_GROUP_COUNTS)
else:
    print('✅ Khớp phân nhóm đã chốt: 47 / 33 / 19 / 15 / 1')
print('Các class thuộc ngoại lệ hiệu lệnh:', classes.loc[classes['code'].isin(HIEU_LENH_EXCEPTIONS), 'class_name'].tolist())
savetab(classes, 'class_group_mapping')

# %% [markdown]
# ## 2. Đọc nhãn và kiểm tra toàn vẹn
#
# **Dòng nhãn dạng đa giác (polygon)** (`class x1 y1 x2 y2 … xn yn`, từ 3 điểm trở lên): được **chuẩn hóa thành bbox** bằng min/max tọa độ (cắt về [0, 1]), giống hệt notebook 03 khi chia tập. Nhờ vậy EDA chạy trên **toàn bộ** ảnh của `cleaned_data.zip` và các con số khớp với bộ dữ liệu đã chia (cập nhật 17/10/2026; bản trước loại 22 ảnh có polygon).
#
# **Ảnh bị loại khỏi EDA** (chỉ khi gặp, trên dữ liệu thật hiện là 0): ảnh hỏng (không mở được), dòng nhãn có số giá trị không hợp lệ (không phải 5, cũng không phải polygon), hoặc giá trị không phải số. Danh sách lưu ở `tables/eda_excluded_images.csv`.
#
# **Box bị bỏ khỏi thống kê** (ảnh vẫn giữ): mỗi dòng nhãn YOLO bắt đầu bằng một số thứ tự class (class id), ứng với vị trí tên biển trong `data.yaml` (0 = `101-duong-cam`, …, 114 = `R.E-11a-bao-hieu-co-ham-chui`). Nếu một dòng ghi số không có trong danh sách (ví dụ 115, 116, thường còn sót lại từ lúc bộ data còn 117 class), notebook không biết đó là biển gì nên không đếm box đó và in cảnh báo kèm danh sách. Bình thường con số này bằng 0.
#
# **Chỉ báo cáo (vẫn giữ):** tọa độ ngoài [0,1], box có cạnh < `TINY_BOX_PX` pixel, box trùng lặp (cùng class, IoU ≥ `DUP_BOX_IOU`), ảnh có cờ xoay EXIF.
#
# Kích thước box theo pixel được tính bằng kích thước ảnh lưu trong file (không áp dụng xoay EXIF), vì bản export không bật preprocessing/auto-orient.

# %%
def read_image_info(path):
    try:
        with Image.open(path) as im:
            w, h = im.size
            orient = im.getexif().get(0x0112, 1)
            im.verify()
        return w, h, orient, ''
    except Exception as e:
        return np.nan, np.nan, np.nan, str(e)[:200]


box_rows, excluded_rows, issue_rows, polygon_rows = [], [], [], []
info = []
for r in images.itertuples():
    w, h, orient, err = read_image_info(MERGED_DIR / 'images' / r.image)
    info.append((w, h, orient))
    reasons = []
    if err:
        reasons.append('anh_hong')
    lines = []
    if r.has_label_file:
        lines = [l for l in (MERGED_DIR / 'labels' / f'{r.stem}.txt').read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()]
    parsed = []
    for ln, line in enumerate(lines, 1):
        parts = line.split()
        try:
            vals = list(map(float, parts))
        except ValueError:
            reasons.append('gia_tri_nhan_khong_phai_so')
            continue
        coords = vals[1:]
        if len(coords) == 4:
            parsed.append((ln, vals))
        elif len(coords) >= 6 and len(coords) % 2 == 0:      # polygon → bbox (giống notebook 03)
            xs, ys = np.clip(coords[0::2], 0, 1), np.clip(coords[1::2], 0, 1)
            x0, x1_, y0, y1_ = xs.min(), xs.max(), ys.min(), ys.max()
            parsed.append((ln, [vals[0], (x0 + x1_) / 2, (y0 + y1_) / 2, x1_ - x0, y1_ - y0]))
            polygon_rows.append({'image': r.image, 'line': ln, 'n_points': len(coords) // 2})
        else:
            reasons.append('so_gia_tri_nhan_khong_hop_le')
    if reasons:
        excluded_rows.append({'image': r.image, 'reasons': ';'.join(sorted(set(reasons)))})
        continue
    for ln, (cid, xc, yc, bw, bh) in parsed:
        if cid != int(cid) or not (0 <= int(cid) < NC):
            issue_rows.append({'image': r.image, 'line': ln, 'issue': 'class_id_ngoai_pham_vi (box bi bo khoi thong ke)', 'value': cid})
            continue
        box_rows.append({'image': r.image, 'line': ln, 'class_id': int(cid), 'xc': xc, 'yc': yc, 'w': bw, 'h': bh})

images[['img_w', 'img_h', 'exif_orientation']] = pd.DataFrame(info, index=images.index)
excluded = pd.DataFrame(excluded_rows, columns=['image', 'reasons'])
savetab(excluded, 'eda_excluded_images')
polygons = pd.DataFrame(polygon_rows, columns=['image', 'line', 'n_points'])
savetab(polygons, 'polygon_lines_converted_to_bbox')
print(f'Đã chuẩn hóa {len(polygons)} dòng polygon thành bbox (trong {polygons["image"].nunique()} ảnh)')
images['in_eda'] = ~images['image'].isin(excluded['image'])

boxes = pd.DataFrame(box_rows)
boxes = boxes.merge(images[['image', 'img_w', 'img_h']], on='image')
boxes = boxes.merge(classes[['class_id', 'class_name', 'code', 'group']], on='class_id')
boxes['w_px'] = boxes['w'] * boxes['img_w']
boxes['h_px'] = boxes['h'] * boxes['img_h']
boxes['area_px'] = boxes['w_px'] * boxes['h_px']
boxes['sqrt_area_px'] = np.sqrt(boxes['area_px'])
boxes['rel_area'] = boxes['w'] * boxes['h']
boxes['aspect_wh'] = boxes['w_px'] / boxes['h_px'].replace(0, np.nan)
boxes['size_bin'] = pd.cut(boxes['area_px'], [-np.inf, COCO_SMALL, COCO_MEDIUM, np.inf],
                           labels=['small (<32²)', 'medium (32²–96²)', 'large (≥96²)'], right=False)
boxes = boxes.reset_index(drop=True)
boxes['box_id'] = boxes.index

# Các vấn đề chỉ báo cáo
out_range = boxes[(boxes[['xc', 'yc', 'w', 'h']] < 0).any(axis=1) | (boxes[['xc', 'yc', 'w', 'h']] > 1).any(axis=1)]
x1, x2 = boxes['xc'] - boxes['w'] / 2, boxes['xc'] + boxes['w'] / 2
y1, y2 = boxes['yc'] - boxes['h'] / 2, boxes['yc'] + boxes['h'] / 2
out_frame = boxes[(x1 < -1e-6) | (y1 < -1e-6) | (x2 > 1 + 1e-6) | (y2 > 1 + 1e-6)]
tiny = boxes[(boxes['w_px'] < TINY_BOX_PX) | (boxes['h_px'] < TINY_BOX_PX)]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0


dup_rows = []
for (img, cid), g in boxes.groupby(['image', 'class_id']):
    if len(g) < 2:
        continue
    vals = g[['xc', 'yc', 'w', 'h']].values
    ids = g['line'].values
    for i, j in combinations(range(len(g)), 2):
        v = iou(vals[i], vals[j])
        if v >= DUP_BOX_IOU:
            dup_rows.append({'image': img, 'class_name': CLASS_NAMES[cid], 'line_a': ids[i], 'line_b': ids[j], 'iou': round(v, 4)})
dup_boxes = pd.DataFrame(dup_rows, columns=['image', 'class_name', 'line_a', 'line_b', 'iou'])
exif_rot = images[images['in_eda'] & images['exif_orientation'].fillna(1).ne(1)]
issues = pd.DataFrame(issue_rows, columns=['image', 'line', 'issue', 'value'])

integrity = pd.DataFrame([
    ('Ảnh trong cleaned_data.zip', len(images)),
    ('Ảnh không có file nhãn (phải = 0 sau clean)', int((~images['has_label_file']).sum())),
    ('Ảnh bị loại khỏi EDA: ảnh hỏng', int(excluded['reasons'].str.contains('anh_hong').sum())),
    ('Dòng polygon đã chuẩn hóa thành bbox', len(polygons)),
    ('Ảnh có dòng polygon (vẫn giữ trong EDA)', int(polygons['image'].nunique())),
    ('Ảnh bị loại khỏi EDA: số giá trị nhãn không hợp lệ', int(excluded['reasons'].str.contains('khong_hop_le').sum())),
    ('Ảnh bị loại khỏi EDA: giá trị nhãn không phải số', int(excluded['reasons'].str.contains('khong_phai_so').sum())),
    ('Ảnh dùng cho EDA', int(images['in_eda'].sum())),
    ('Box có class id ngoài phạm vi (bỏ khỏi thống kê)', len(issues)),
    ('Box có tọa độ tâm/kích thước ngoài [0,1]', len(out_range)),
    ('Box có cạnh vượt ra ngoài khung ảnh', len(out_frame)),
    (f'Box có cạnh < {TINY_BOX_PX} px', len(tiny)),
    (f'Cặp box trùng lặp (cùng class, IoU ≥ {DUP_BOX_IOU})', len(dup_boxes)),
    ('Ảnh có cờ xoay EXIF ≠ 1', len(exif_rot)),
], columns=['kiểm tra', 'số lượng'])
display(integrity)
savetab(integrity, 'integrity_summary')
savetab(issues, 'integrity_class_id_out_of_range')
savetab(out_range, 'integrity_coords_out_of_01')
savetab(out_frame, 'integrity_box_outside_frame')
savetab(tiny, 'integrity_tiny_boxes')
savetab(dup_boxes, 'integrity_duplicate_boxes')
savetab(exif_rot[['image', 'exif_orientation']], 'integrity_exif_rotated')
if len(issues):
    print(f'⚠️ CẢNH BÁO: {len(issues)} box có class id ngoài phạm vi, đã bỏ khỏi thống kê. Xem tables/integrity_class_id_out_of_range.csv')

# Từ đây trở đi chỉ dùng ảnh trong EDA
img_eda = images[images['in_eda']].copy()

# %% [markdown]
# ## 3. Thống kê tổng quan

# %%
n_cls_present = boxes['class_id'].nunique()
overview = pd.DataFrame([
    ('Số ảnh', len(img_eda)),
    ('Số bounding box', len(boxes)),
    ('Số class trong data.yaml', NC),
    ('Số class có ít nhất 1 box', n_cls_present),
    ('Số box trung bình / ảnh', round(len(boxes) / max(len(img_eda), 1), 2)),
    ('Số box trung vị / class', float(boxes['class_id'].value_counts().median())),
], columns=['chỉ số', 'giá trị'])
display(overview)
savetab(overview, 'overview')
missing_cls = classes[~classes['class_id'].isin(boxes['class_id'])]
if len(missing_cls):
    print('⚠️ Các class không có box nào:', missing_cls['class_name'].tolist())

# %% [markdown]
# ## 4. Phân bố class
#
# Số box và số ảnh của mỗi class. Trục số lượng dùng **thang log** để thấy rõ phần đuôi dài.
# Chia nhóm theo số box: **head ≥ 100**, **medium 20–99**, **tail < 20**.

# %%
cls_stats = classes.copy()
cls_stats['n_boxes'] = cls_stats['class_id'].map(boxes['class_id'].value_counts()).fillna(0).astype(int)
cls_stats['n_images'] = cls_stats['class_id'].map(boxes.groupby('class_id')['image'].nunique()).fillna(0).astype(int)
cls_stats['pct_boxes'] = (cls_stats['n_boxes'] / cls_stats['n_boxes'].sum() * 100).round(3)
cls_stats['frequency_tier'] = np.select([cls_stats['n_boxes'] >= HEAD_MIN, cls_stats['n_boxes'] >= TAIL_MAX],
                                        [f'head (≥{HEAD_MIN})', f'medium ({TAIL_MAX}–{HEAD_MIN - 1})'], f'tail (<{TAIL_MAX})')
cls_stats = cls_stats.sort_values('n_boxes', ascending=False).reset_index(drop=True)
savetab(cls_stats, 'class_distribution')

tiers = cls_stats.groupby('frequency_tier').agg(n_classes=('class_id', 'size'), n_boxes=('n_boxes', 'sum'))
tiers['pct_boxes'] = (tiers['n_boxes'] / tiers['n_boxes'].sum() * 100).round(2)
display(tiers)
savetab(tiers, 'class_frequency_tiers', index=True)
imb = cls_stats.loc[cls_stats['n_boxes'] > 0, 'n_boxes']
print(f'Tỉ lệ mất cân bằng (class nhiều nhất / ít nhất, chỉ tính class có box): {imb.max() / imb.min():.1f}')

fig, ax = plt.subplots(figsize=(9, max(6, 0.17 * NC)))
y = np.arange(len(cls_stats))
ax.barh(y, cls_stats['n_boxes'].clip(lower=0.8), color=cls_stats['group'].map(GROUP_COLOR), height=0.75)
ax.set_yticks(y, cls_stats['class_name'], fontsize=6.5)
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlabel('Số bounding box (thang log)')
ax.set_title('Số box theo class (sắp xếp giảm dần)')
for v in (HEAD_MIN, TAIL_MAX):
    ax.axvline(v, color='#8a8984', ls='--', lw=0.8)
ax.legend(handles=[Patch(color=GROUP_COLOR[g], label=g) for g in GROUP_ORDER], loc='lower right', frameon=False)
ax.grid(axis='y', visible=False)
savefig(fig, '04_class_distribution_boxes')

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.arange(1, len(cls_stats) + 1), cls_stats['n_boxes'].clip(lower=0.8), color='#2a78d6', lw=2)
ax.set_yscale('log')
ax.set_xlabel('Thứ hạng class')
ax.set_ylabel('Số box (log)')
ax.set_title('Đường cong đuôi dài của phân bố class')
savefig(fig, '04_class_long_tail_curve')

print(f'Các class tail (<{TAIL_MAX} box):')
display(cls_stats.loc[cls_stats['n_boxes'] < TAIL_MAX, ['class_name', 'group', 'n_boxes', 'n_images']])

# %% [markdown]
# ## 5. Phân bố theo 5 nhóm biển báo

# %%
grp = classes.groupby('group').size().rename('n_classes').to_frame()
grp['n_boxes'] = boxes.groupby('group').size()
grp['n_images'] = boxes.groupby('group')['image'].nunique()
grp = grp.reindex(GROUP_ORDER).fillna(0).astype(int)
grp['pct_boxes'] = (grp['n_boxes'] / grp['n_boxes'].sum() * 100).round(2)
grp['boxes_per_class'] = (grp['n_boxes'] / grp['n_classes']).round(1)
display(grp)
savetab(grp, 'group_distribution', index=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(axes, ['n_classes', 'n_boxes', 'n_images'], ['Số class', 'Số box', 'Số ảnh chứa nhóm']):
    bars = ax.bar(GROUP_ORDER, grp[col], color=[GROUP_COLOR[g] for g in GROUP_ORDER], width=0.65)
    ax.bar_label(bars, fmt='%d', fontsize=9, color='#0b0b0b')
    ax.set_title(title)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', rotation=20)
fig.suptitle('Phân bố theo nhóm biển báo (QCVN 41:2024)', fontweight='bold')
savefig(fig, '05_group_distribution')

fig, ax = plt.subplots(figsize=(10, 4))
for i, g in enumerate(GROUP_ORDER):
    v = cls_stats.loc[cls_stats['group'] == g, 'n_boxes'].clip(lower=0.8)
    ax.scatter(np.random.normal(i, 0.07, len(v)), v, s=22, color=GROUP_COLOR[g], alpha=0.85, edgecolor='white', linewidth=0.6)
ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER)
ax.set_yscale('log')
ax.set_ylabel('Số box mỗi class (log)')
ax.set_title('Số box của từng class trong mỗi nhóm')
ax.grid(axis='x', visible=False)
savefig(fig, '05_boxes_per_class_by_group')

# %% [markdown]
# ## 6. Số box mỗi ảnh

# %%
bpi = boxes.groupby('image').size().reindex(img_eda['image'], fill_value=0)
img_eda['n_boxes'] = img_eda['image'].map(bpi)
img_eda['n_classes'] = img_eda['image'].map(boxes.groupby('image')['class_id'].nunique()).fillna(0).astype(int)
bpi_stats = pd.DataFrame([
    ('Ảnh có 0 box hợp lệ', int((bpi == 0).sum())),
    ('Ảnh có đúng 1 box', int((bpi == 1).sum())),
    ('Ảnh có ≥ 2 box', int((bpi >= 2).sum())),
    ('Ảnh có ≥ 2 class khác nhau', int((img_eda['n_classes'] >= 2).sum())),
    ('Số box nhiều nhất trong 1 ảnh', int(bpi.max())),
    ('Trung bình box/ảnh', round(bpi.mean(), 2)),
    ('Trung vị box/ảnh', float(bpi.median())),
], columns=['chỉ số', 'giá trị'])
display(bpi_stats)
savetab(bpi_stats, 'boxes_per_image_summary')
savetab(bpi.value_counts().sort_index().rename_axis('n_boxes').rename('n_images').reset_index(), 'boxes_per_image_hist')

fig, ax = plt.subplots(figsize=(9, 3.8))
vc = bpi.value_counts().sort_index()
ax.bar(vc.index, vc.values, color='#2a78d6', width=0.8)
ax.set_xlabel('Số box trong ảnh')
ax.set_ylabel('Số ảnh')
ax.set_title('Phân bố số box mỗi ảnh')
ax.grid(axis='x', visible=False)
savefig(fig, '06_boxes_per_image')

# %% [markdown]
# ## 7. Kích thước ảnh

# %%
img_eda['resolution'] = img_eda['img_w'].astype(int).astype(str) + '×' + img_eda['img_h'].astype(int).astype(str)
img_eda['aspect'] = img_eda['img_w'] / img_eda['img_h']
res = img_eda['resolution'].value_counts().rename_axis('resolution').rename('n_images').reset_index()
res['pct'] = (res['n_images'] / len(img_eda) * 100).round(2)
print('Số độ phân giải khác nhau:', len(res))
display(res.head(15))
savetab(res, 'image_resolutions')
display(img_eda[['img_w', 'img_h', 'aspect']].describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cnt = img_eda.groupby(['img_w', 'img_h']).size().reset_index(name='n')
axes[0].scatter(cnt['img_w'], cnt['img_h'], s=20 + 300 * cnt['n'] / cnt['n'].max(), color='#2a78d6', alpha=0.7, edgecolor='white')
axes[0].set_xlabel('Chiều rộng (px)')
axes[0].set_ylabel('Chiều cao (px)')
axes[0].set_title('Độ phân giải ảnh (kích thước điểm ∝ số ảnh)')
axes[1].hist(img_eda['aspect'], bins=40, color='#2a78d6', edgecolor='white')
axes[1].set_xlabel('Tỉ lệ khung hình W/H')
axes[1].set_ylabel('Số ảnh')
axes[1].set_title('Tỉ lệ khung hình ảnh')
savefig(fig, '07_image_sizes')

# %% [markdown]
# ## 8. Kích thước bounding box
#
# Theo pixel, theo diện tích tương đối so với ảnh, và theo chuẩn COCO: **small < 32²**, **medium 32²–96²**, **large ≥ 96²** pixel.

# %%
size_desc = boxes[['w_px', 'h_px', 'sqrt_area_px', 'rel_area']].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(4)
display(size_desc)
savetab(size_desc, 'box_size_describe', index=True)

coco = boxes['size_bin'].value_counts().reindex(boxes['size_bin'].cat.categories).rename('n_boxes').to_frame()
coco['pct'] = (coco['n_boxes'] / coco['n_boxes'].sum() * 100).round(2)
display(coco)
savetab(coco, 'box_size_coco_bins', index=True)
coco_grp = pd.crosstab(boxes['group'], boxes['size_bin'], normalize='index').reindex(GROUP_ORDER).fillna(0) * 100
savetab(coco_grp.round(2), 'box_size_coco_bins_by_group', index=True)
coco_cls = pd.crosstab(boxes['class_name'], boxes['size_bin'])
savetab(coco_cls, 'box_size_coco_bins_by_class', index=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
bins = np.logspace(np.log10(max(boxes['sqrt_area_px'].min(), 1)), np.log10(boxes['sqrt_area_px'].max() + 1), 50)
axes[0].hist(boxes['sqrt_area_px'], bins=bins, color='#2a78d6', edgecolor='white')
axes[0].set_xscale('log')
for v in (32, 96):
    axes[0].axvline(v, color='#8a8984', ls='--', lw=0.8)
axes[0].set_xlabel('√(diện tích box) (px, log)')
axes[0].set_ylabel('Số box')
_t = [t for t in (4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048) if bins[0] <= t <= bins[-1]]
axes[0].set_xticks(_t, [str(t) for t in _t])
axes[0].minorticks_off()
axes[0].set_title('Kích thước box theo pixel')
rb = np.logspace(np.log10(max(boxes['rel_area'].min(), 1e-7)), np.log10(boxes['rel_area'].max()), 50)
axes[1].hist(boxes['rel_area'], bins=rb, color='#2a78d6', edgecolor='white')
axes[1].set_xscale('log')
axes[1].set_xlabel('Diện tích box / diện tích ảnh (log)')
axes[1].set_title('Diện tích tương đối')
seq = plt.get_cmap(SEQ_CMAP)([0.35, 0.6, 0.85])
left = np.zeros(len(GROUP_ORDER))
for k, c in zip(coco_grp.columns, seq):
    axes[2].barh(GROUP_ORDER, coco_grp[k], left=left, color=c, label=k, edgecolor='white', linewidth=2)
    left += coco_grp[k].values
axes[2].invert_yaxis()
axes[2].set_xlabel('% box trong nhóm')
axes[2].set_title('Tỉ lệ small/medium/large theo nhóm')
axes[2].legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
axes[2].grid(axis='y', visible=False)
savefig(fig, '08_box_sizes')

fig, ax = plt.subplots(figsize=(9, 4))
data = [boxes.loc[boxes['group'] == g, 'sqrt_area_px'] for g in GROUP_ORDER]
bp = ax.boxplot(data, tick_labels=GROUP_ORDER, patch_artist=True, showfliers=False, widths=0.55)
for patch, g in zip(bp['boxes'], GROUP_ORDER):
    patch.set_facecolor(GROUP_COLOR[g])
    patch.set_alpha(0.8)
for m in bp['medians']:
    m.set_color('#0b0b0b')
ax.set_yscale('log')
ax.set_ylabel('√(diện tích box) (px, log)')
ax.set_title('Kích thước box theo nhóm (không vẽ outlier)')
ax.grid(axis='x', visible=False)
savefig(fig, '08_box_size_by_group')

# %% [markdown]
# ## 9. Tỉ lệ khung box (w/h)
#
# Theo quyết định của nhóm, mục này **chỉ vẽ phân bố**, không lọc bất thường riêng. Box bất thường được xét ở mục 13, dùng cả kích thước và tỉ lệ.

# %%
display(boxes.groupby('group')['aspect_wh'].describe(percentiles=[.05, .5, .95]).reindex(GROUP_ORDER).round(3))
savetab(boxes.groupby('class_name')['aspect_wh'].describe(percentiles=[.05, .5, .95]).round(3), 'box_aspect_by_class', index=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ab = np.logspace(np.log10(boxes['aspect_wh'].quantile(0.001)), np.log10(boxes['aspect_wh'].quantile(0.999)), 60)
axes[0].hist(boxes['aspect_wh'].clip(ab[0], ab[-1]), bins=ab, color='#2a78d6', edgecolor='white')
axes[0].set_xscale('log')
axes[0].axvline(1, color='#8a8984', ls='--', lw=0.8)
axes[0].set_xlabel('w/h (log)')
axes[0].set_ylabel('Số box')
axes[0].set_title('Tỉ lệ khung box – toàn bộ')
for g in GROUP_ORDER:
    v = boxes.loc[boxes['group'] == g, 'aspect_wh'].clip(ab[0], ab[-1])
    if len(v):
        axes[1].hist(v, bins=ab, histtype='step', lw=2, color=GROUP_COLOR[g], label=g, density=True)
axes[1].set_xscale('log')
axes[1].set_xlabel('w/h (log)')
axes[1].set_ylabel('Mật độ')
axes[1].set_title('Tỉ lệ khung box theo nhóm')
axes[1].legend(frameon=False)
savefig(fig, '09_box_aspect')

# %% [markdown]
# ## 10. Heatmap vị trí tâm box trong khung hình
#
# Tọa độ chuẩn hóa (0–1), gốc ở góc trên trái như trong ảnh.

# %%
def center_heatmap(ax, df, title, bins=50):
    hm, _, _ = np.histogram2d(df['yc'].clip(0, 1), df['xc'].clip(0, 1), bins=bins, range=[[0, 1], [0, 1]])
    im = ax.imshow(hm, cmap=SEQ_CMAP, extent=[0, 1, 1, 0], aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    return im


fig, ax = plt.subplots(figsize=(6, 4.5))
im = center_heatmap(ax, boxes, f'Vị trí tâm box (n = {len(boxes)})')
ax.set_xlabel('x (chuẩn hóa)')
ax.set_ylabel('y (chuẩn hóa)')
fig.colorbar(im, ax=ax, label='Số box')
savefig(fig, '10_center_heatmap')

fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(18, 3.6))
for ax, g in zip(axes, GROUP_ORDER):
    sub = boxes[boxes['group'] == g]
    im = center_heatmap(ax, sub, f'{g} (n = {len(sub)})', bins=30)
    fig.colorbar(im, ax=ax, fraction=0.046)
savefig(fig, '10_center_heatmap_by_group')

# %% [markdown]
# ## 11. Ma trận đồng xuất hiện
#
# Ô (A, B) = số ảnh chứa đồng thời A và B. Đường chéo = số ảnh chứa A.
# Bảng đầy đủ 115 × 115 lưu ở `tables/cooccurrence_class.csv`; biểu đồ chỉ vẽ 30 class nhiều ảnh nhất.

# %%
img_cls = boxes.groupby('image')['class_id'].apply(lambda s: sorted(set(s)))
co = np.zeros((NC, NC), dtype=int)
for cl in img_cls:
    for a in cl:
        co[a, a] += 1
    for a, b in combinations(cl, 2):
        co[a, b] += 1
        co[b, a] += 1
co_df = pd.DataFrame(co, index=CLASS_NAMES, columns=CLASS_NAMES)
savetab(co_df, 'cooccurrence_class', index=True)

pairs = [(CLASS_NAMES[a], CLASS_NAMES[b], co[a, b]) for a in range(NC) for b in range(a + 1, NC) if co[a, b] > 0]
pairs = pd.DataFrame(pairs, columns=['class_a', 'class_b', 'n_images']).sort_values('n_images', ascending=False)
savetab(pairs, 'cooccurrence_pairs')
print('Các cặp class đồng xuất hiện nhiều nhất:')
display(pairs.head(15))

img_grp = boxes.groupby('image')['group'].apply(lambda s: sorted(set(s)))
gco = pd.DataFrame(0, index=GROUP_ORDER, columns=GROUP_ORDER)
for gl in img_grp:
    for a in gl:
        gco.loc[a, a] += 1
    for a, b in combinations(gl, 2):
        gco.loc[a, b] += 1
        gco.loc[b, a] += 1
savetab(gco, 'cooccurrence_group', index=True)

top = cls_stats.sort_values('n_images', ascending=False).head(30)['class_name'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 2.2]})
im = axes[0].imshow(gco.values, cmap=SEQ_CMAP)
axes[0].set_xticks(range(5), GROUP_ORDER, rotation=30, ha='right')
axes[0].set_yticks(range(5), GROUP_ORDER)
for i in range(5):
    for j in range(5):
        v = gco.values[i, j]
        axes[0].text(j, i, v, ha='center', va='center', fontsize=9, color='white' if v > gco.values.max() * 0.6 else '#0b0b0b')
axes[0].set_title('Đồng xuất hiện giữa các nhóm (số ảnh)')
axes[0].grid(False)
sub = co_df.loc[top, top].values.astype(float)
np.fill_diagonal(sub, np.nan)
im2 = axes[1].imshow(sub, cmap=SEQ_CMAP)
axes[1].set_xticks(range(len(top)), top, rotation=90, fontsize=6.5)
axes[1].set_yticks(range(len(top)), top, fontsize=6.5)
axes[1].set_title('Đồng xuất hiện giữa 30 class nhiều ảnh nhất (bỏ đường chéo)')
axes[1].grid(False)
fig.colorbar(im2, ax=axes[1], fraction=0.03, label='Số ảnh')
savefig(fig, '11_cooccurrence')

# %% [markdown]
# ## 12. Ảnh mẫu cho mỗi class
#
# Mỗi class lấy ngẫu nhiên tối đa `N_SAMPLES_PER_CLASS` ảnh (seed cố định). Box của class đang xét được vẽ đậm theo màu nhóm, box của class khác vẽ mảnh màu xám.
# Mỗi class lưu thành một file trong `samples_per_class/`, đồng thời hiển thị ngay dưới đây.

# %%
def load_rgb(name, max_side=800):
    im = Image.open(MERGED_DIR / 'images' / name).convert('RGB')
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)))
    return im


def safe_name(s):
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in s)


by_image = {k: g for k, g in boxes.groupby('image')}
for _, c in classes.iterrows():
    imgs = sorted(boxes.loc[boxes['class_id'] == c['class_id'], 'image'].unique())
    if not imgs:
        continue
    pick = random.Random(SEED + int(c['class_id'])).sample(imgs, min(N_SAMPLES_PER_CLASS, len(imgs)))
    fig, axes = plt.subplots(1, N_SAMPLES_PER_CLASS, figsize=(4 * N_SAMPLES_PER_CLASS, 3.2))
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.axis('off')
    for ax, name in zip(axes, pick):
        im = load_rgb(name)
        ax.imshow(im)
        for b in by_image[name].itertuples():
            x0, y0 = (b.xc - b.w / 2) * im.width, (b.yc - b.h / 2) * im.height
            is_t = b.class_id == c['class_id']
            ax.add_patch(Rectangle((x0, y0), b.w * im.width, b.h * im.height, fill=False,
                                   ec=GROUP_COLOR[c['group']] if is_t else '#bdbdbd', lw=2.2 if is_t else 0.8))
        ax.set_title(name[:38], fontsize=7)
    n_b = int((boxes['class_id'] == c['class_id']).sum())
    fig.suptitle(f"[{c['class_id']}] {c['class_name']}  |  nhóm {c['group']}  |  {n_b} box, {len(imgs)} ảnh", fontsize=11, fontweight='bold')
    fig.savefig(SAMPLE_DIR / f"{c['class_id']:03d}_{safe_name(c['class_name'])}.png", dpi=110)
    plt.show()

# %% [markdown]
# ### 12b. Ảnh crop từng biển theo class
#
# - `crops_per_class/<class>.png`: tối đa `N_CROPS_PER_CLASS` crop ngẫu nhiên mỗi class, để 3 thành viên soát nhầm nhãn bằng mắt.
# - Hình tổng quan 115 lớp cho bài báo: xem mục 12c (chọn được ảnh đại diện bằng tay).

# %%
def crop_box(b, pad=0.1, size=96):
    with Image.open(MERGED_DIR / 'images' / b.image) as im:
        im = im.convert('RGB')
        W, H = im.size
        bw, bh = b.w * W, b.h * H
        x0, y0 = (b.xc * W - bw / 2) - pad * bw, (b.yc * H - bh / 2) - pad * bh
        x1, y1 = (b.xc * W + bw / 2) + pad * bw, (b.yc * H + bh / 2) + pad * bh
        box = (max(0, int(x0)), max(0, int(y0)), min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1))))
        if box[2] <= box[0] or box[3] <= box[1]:
            return Image.new('RGB', (size, size), (230, 230, 230))
        return ImageOps.pad(im.crop(box), (size, size), color=(245, 245, 245))


for _, c in classes.iterrows():
    sub = boxes[boxes['class_id'] == c['class_id']]
    if sub.empty:
        continue
    sub = sub.sample(min(N_CROPS_PER_CLASS, len(sub)), random_state=SEED)
    ncol = 8
    nrow = int(np.ceil(len(sub) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.3, nrow * 1.45))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, b in zip(axes, sub.itertuples()):
        ax.imshow(crop_box(b))
        ax.set_title(f'{int(b.w_px)}×{int(b.h_px)}', fontsize=6)
    fig.suptitle(f"[{c['class_id']}] {c['class_name']} ({c['group']})", fontsize=9, fontweight='bold')
    fig.savefig(CROP_DIR / f"{c['class_id']:03d}_{safe_name(c['class_name'])}.png", dpi=110)
    plt.close(fig)
print('Đã lưu crop của từng class vào', CROP_DIR)

# %% [markdown]
# ### 12c. Hình tổng quan 115 lớp – chọn ảnh đại diện bằng tay
#
# Mặc định mỗi lớp lấy **box lớn nhất** (ứng viên #1). Nếu box đó bị gán nhãn sai hoặc ảnh xấu, nhóm chọn ứng viên khác:
#
# 1. Chạy notebook một lần. Xem các trang ứng viên trong `output_eda/montage_ung_vien/` (mỗi lớp một trang, crop đánh số #1, #2, …;
#    danh sách đầy đủ kèm tên ảnh ở `tables/montage_ung_vien.csv`).
# 2. Mở `output_eda/montage_chon_anh_MAU.csv` (đã có sẵn 115 dòng), lưu thành **`output_eda/montage_chon_anh.csv`** và điền:
#    - cột `chon`: số thứ tự ứng viên muốn dùng (để trống = giữ #1);
#    - cột `sai_nhan`: các số thứ tự bị **gán nhãn sai** (cách nhau bằng dấu `;`, ví dụ `1;4`) – để ghi lại, sửa trên Roboflow ở phiên bản sau.
#    Chỉ cần điền các lớp muốn đổi; dòng để trống giữ mặc định.
# 3. Chạy lại notebook (hoặc chỉ chạy lại từ mục 12b). Hình `figures/12_class_overview_montage.png` dùng ảnh đã chọn;
#    các box bị đánh dấu sai nhãn được ghi vào `tables/loi_nhan_bao_cao.csv`.
#
# Việc chọn ảnh chỉ thay đổi hình minh họa, không thay đổi dữ liệu hay bất kỳ con số nào khác của EDA.

# %%
CAND_DIR.mkdir(parents=True, exist_ok=True)
cand = (boxes.sort_values(['class_id', 'area_px'], ascending=[True, False])
        .groupby('class_id', group_keys=False).head(MONTAGE_N_CANDIDATES).copy())
cand['idx'] = cand.groupby('class_id').cumcount() + 1
cand = cand.merge(classes[['class_id', 'short']], on='class_id')
savetab(cand[['class_id', 'short', 'class_name', 'idx', 'image', 'line', 'w_px', 'h_px']], 'montage_ung_vien')

for cid, sub in cand.groupby('class_id'):
    ncol = 8
    nrow = int(np.ceil(len(sub) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.4, nrow * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, b in zip(axes, sub.itertuples()):
        ax.imshow(crop_box(b, size=128))
        ax.set_title(f'#{b.idx}  {int(b.w_px)}×{int(b.h_px)}', fontsize=7)
    r0 = sub.iloc[0]
    fig.suptitle(f"[{cid}] {r0['class_name']} – ứng viên cho hình tổng quan", fontsize=9, fontweight='bold')
    fig.savefig(CAND_DIR / f"{cid:03d}_{safe_name(r0['class_name'])}.png", dpi=110)
    plt.close(fig)

tmpl = classes[['class_id', 'short', 'class_name']].copy()
tmpl['so_ung_vien'] = tmpl['class_id'].map(cand.groupby('class_id').size()).fillna(0).astype(int)
tmpl['chon'] = ''
tmpl['sai_nhan'] = ''
tmpl.to_csv(EDA_OUT / 'montage_chon_anh_MAU.csv', index=False, encoding='utf-8-sig')
print(f'Đã lưu trang ứng viên vào {CAND_DIR} và file mẫu montage_chon_anh_MAU.csv')

choice, reported = {}, []
if MONTAGE_CHOICE_CSV.exists():
    ch = pd.read_csv(MONTAGE_CHOICE_CSV, encoding='utf-8-sig', dtype=str).fillna('')
    short2cid = dict(zip(classes['short'].astype(str), classes['class_id']))
    for r in ch.itertuples():
        cid = short2cid.get(str(r.short).strip())
        if cid is None:
            raise ValueError(f'montage_chon_anh.csv: không có lớp "{r.short}"')
        n_c = int((cand['class_id'] == cid).sum())
        if str(r.chon).strip():
            k = int(float(r.chon))
            if not 1 <= k <= n_c:
                raise ValueError(f'montage_chon_anh.csv: lớp {r.short} chỉ có {n_c} ứng viên, không có #{k}')
            choice[cid] = k
        for t in str(r.sai_nhan).replace(',', ';').split(';'):
            if t.strip():
                k = int(float(t))
                if not 1 <= k <= n_c:
                    raise ValueError(f'montage_chon_anh.csv: lớp {r.short} không có ứng viên #{k} (cột sai_nhan)')
                reported.append((cid, k))
    print(f'Đọc {MONTAGE_CHOICE_CSV.name}: đổi ảnh cho {len(choice)} lớp, ghi nhận {len(reported)} box sai nhãn')
    bad = {(c, k) for c, k in reported}
    for cid, k in choice.items():
        if (cid, k) in bad:
            print(f'⚠️ Lớp {cid}: ứng viên được chọn #{k} cũng nằm trong cột sai_nhan')
else:
    print('Chưa có montage_chon_anh.csv → mọi lớp dùng ứng viên #1 (box lớn nhất)')

if reported:
    rp = pd.DataFrame(reported, columns=['class_id', 'idx']).merge(cand, on=['class_id', 'idx'])
    savetab(rp[['class_id', 'short', 'class_name', 'idx', 'image', 'line', 'w_px', 'h_px']], 'loi_nhan_bao_cao')

cand['chosen'] = cand.apply(lambda r: r['idx'] == choice.get(r['class_id'], 1), axis=1)
rep = cand[cand['chosen']].copy()
savetab(rep[['class_id', 'short', 'class_name', 'idx', 'image', 'line']], 'montage_anh_da_chon')
rep = rep.assign(go=rep['group'].map({g: i for i, g in enumerate(GROUP_ORDER)})).sort_values(['go', 'class_id'])
ncol = 12
nrow = int(np.ceil(len(rep) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.35, nrow * 1.75))
axes = np.atleast_1d(axes).ravel()
for ax in axes:
    ax.axis('off')
for ax, b in zip(axes, rep.itertuples()):
    ax.imshow(crop_box(b, size=128))
    ax.set_title(b.short, fontsize=7, color='#0b0b0b')
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(GROUP_COLOR[b.group])
        s.set_linewidth(2.5)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(top=1 - 0.6 / (nrow * 1.75), bottom=0.9 / (nrow * 1.75), hspace=0.35)
fig.legend(handles=[Patch(facecolor='white', edgecolor=GROUP_COLOR[g], linewidth=2.5, label=g) for g in GROUP_ORDER],
           loc='lower center', ncol=5, frameon=False)
fig.suptitle('Một mẫu đại diện cho mỗi class (viền màu = nhóm biển)', fontweight='bold', y=1 - 0.15 / (nrow * 1.75))
savefig(fig, '12_class_overview_montage')

# %% [markdown]
# ## 13. Box bất thường theo từng class
#
# Với mỗi class có ít nhất `ANOM_MIN_BOXES` box, tính **robust z-score** (dùng median và MAD) cho hai đặc trưng:
# - kích thước: log(√diện tích box theo pixel)
# - tỉ lệ: log(w/h)
#
# Box có |z| > `ANOM_Z` ở ít nhất một đặc trưng bị đánh dấu. Danh sách xuất ra `tables/anomalous_boxes.csv` để 3 thành viên soát lại. **Notebook không sửa hay xóa nhãn nào.**
# Box lệch về kích thước có thể chỉ là biển ở rất xa hoặc rất gần, không nhất thiết là nhãn sai.

# %%
def robust_z(x):
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return np.zeros_like(x) if mad == 0 else (x - med) / mad


anom_parts = []
for cid, g in boxes.groupby('class_id'):
    if len(g) < ANOM_MIN_BOXES:
        continue
    g = g.copy()
    g['z_size'] = robust_z(np.log(g['sqrt_area_px'].clip(lower=1e-3).values))
    g['z_aspect'] = robust_z(np.log(g['aspect_wh'].clip(lower=1e-3).values))
    anom_parts.append(g)
scored = pd.concat(anom_parts) if anom_parts else boxes.iloc[:0].assign(z_size=[], z_aspect=[])
scored['flag_size'] = scored['z_size'].abs() > ANOM_Z
scored['flag_aspect'] = scored['z_aspect'].abs() > ANOM_Z
anom = scored[scored['flag_size'] | scored['flag_aspect']].copy()
anom['max_abs_z'] = anom[['z_size', 'z_aspect']].abs().max(axis=1)
anom = anom.sort_values('max_abs_z', ascending=False)
cols = ['image', 'line', 'class_name', 'group', 'w_px', 'h_px', 'aspect_wh', 'z_size', 'z_aspect', 'flag_size', 'flag_aspect']
savetab(anom[cols].round(3), 'anomalous_boxes')
n_skip = int((cls_stats['n_boxes'].between(1, ANOM_MIN_BOXES - 1)).sum())
print(f'Số box được xét: {len(scored)} (bỏ qua {n_skip} class có < {ANOM_MIN_BOXES} box)')
print(f"Box bất thường: {len(anom)} (kích thước: {int(anom['flag_size'].sum())}, tỉ lệ: {int(anom['flag_aspect'].sum())})")
display(anom[cols].head(15).round(2))
display(anom.groupby('class_name').size().sort_values(ascending=False).head(15).rename('số box bất thường').to_frame())

show = anom.head(48)
if len(show):
    ncol = 8
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 2.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, b in zip(axes, show.itertuples()):
        ax.imshow(crop_box(b, pad=0.25))
        tag = ('S' if b.flag_size else '') + ('A' if b.flag_aspect else '')
        ax.set_title(f'{b.class_name.split("-")[0]} [{tag}]\n{int(b.w_px)}×{int(b.h_px)}', fontsize=6.5)
    fig.suptitle('48 box bất thường nhất (S = kích thước, A = tỉ lệ w/h)', fontweight='bold')
    savefig(fig, '13_anomalous_boxes')

# %% [markdown]
# ## 14. Xuất bảng tổng hợp
#
# - `tables/images.csv`: mỗi dòng là một ảnh (kích thước, số box, số class)
# - `tables/boxes.csv`: mỗi dòng là một box (class, nhóm, tọa độ, kích thước pixel, bin COCO)
# - `summary.json`: các con số chính để đưa vào bài báo

# %%
savetab(img_eda.drop(columns=['has_label_file', 'in_eda']), 'images')
savetab(boxes.drop(columns=['box_id']), 'boxes')

summary = {
    'dataset': 'Roboflow dataset-dap391m v4 (YOLOv8), sau notebook 01_clean_data',
    'n_images_in_cleaned_zip': int(len(images)),
    'n_images_excluded_from_eda': int(len(excluded)),
    'n_polygon_lines_converted': int(len(polygons)),
    'n_images': int(len(img_eda)),
    'n_boxes': int(len(boxes)),
    'n_classes_yaml': NC,
    'n_classes_with_boxes': int(n_cls_present),
    'classes_per_group': {g: int(group_n_classes[g]) for g in GROUP_ORDER},
    'boxes_per_group': {g: int(grp.loc[g, 'n_boxes']) for g in GROUP_ORDER},
    'frequency_tiers': {k: int(v) for k, v in tiers['n_classes'].items()},
    'boxes_per_image_mean': round(float(bpi.mean()), 3),
    'coco_size_bins_pct': {k: float(v) for k, v in coco['pct'].items()},
    'box_sqrt_area_px_median': round(float(boxes['sqrt_area_px'].median()), 2),
    'box_rel_area_median': float(boxes['rel_area'].median()),
    'n_resolutions': int(len(res)),
    'n_anomalous_boxes': int(len(anom)),
    'params': {'HEAD_MIN': HEAD_MIN, 'TAIL_MAX': TAIL_MAX, 'TINY_BOX_PX': TINY_BOX_PX, 'DUP_BOX_IOU': DUP_BOX_IOU,
               'ANOM_MIN_BOXES': ANOM_MIN_BOXES, 'ANOM_Z': ANOM_Z, 'SEED': SEED},
}
with open(EDA_OUT / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))

print('\nĐã lưu vào', EDA_OUT)
for sub in ('figures', 'tables', 'samples_per_class', 'crops_per_class'):
    print(f'  {sub:20s} {len(list((EDA_OUT / sub).iterdir())):4d} file')
