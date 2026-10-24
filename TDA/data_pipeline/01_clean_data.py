# %% [markdown]
# # 01 – Clean data: loại bỏ ảnh không có nhãn
#
# **Đề tài:** VN Traffic Sign (QCVN 41:2024/BGTVT) – bộ dữ liệu Roboflow `dataset-dap391m`, version 4, định dạng YOLOv8.
#
# **Notebook này làm đúng một việc:** loại các ảnh **không có nhãn** và giữ nguyên mọi thứ khác.
#
# | Trường hợp | Xử lý |
# |---|---|
# | Ảnh không có file nhãn `.txt` tương ứng | **Loại** |
# | Ảnh có file nhãn nhưng file rỗng (hoặc chỉ có khoảng trắng) | **Loại** (kèm file nhãn rỗng đó) |
# | Ảnh hỏng, nhãn sai định dạng (polygon, không đủ 5 giá trị), class id ngoài phạm vi, tọa độ ngoài [0,1], file nhãn không có ảnh | **Chỉ báo cáo**, không loại (theo quyết định của nhóm) |
#
# **Đầu vào:** `MyDrive/VNTrafficSign/data/Dataset DAP391m.v4i.yolov8.zip`
#
# **Đầu ra** (thư mục `MyDrive/VNTrafficSign/output_cleaneddata/`):
# - `cleaned_data.zip`: cùng cấu trúc với file zip gốc (train/valid/test, `data.yaml`, README), chỉ thiếu các ảnh bị loại
# - `removed_images.csv`: danh sách ảnh bị loại và lý do
# - `clean_summary.csv`: số ảnh trước/sau theo từng split
# - `report_*.csv`: các vấn đề chỉ báo cáo (không loại)
#
# **Dung lượng Drive:** file zip được giải nén vào ổ cục bộ `/content` của Colab, không giải nén lên Drive. Chỉ `cleaned_data.zip` và các file CSV được ghi lên Drive.

# %% [markdown]
# ## 1. Cấu hình và kết nối Google Drive

# %%
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive/VNTrafficSign')
    LOCAL_ROOT = Path('/content')
except ImportError:  # chạy ngoài Colab (dùng để kiểm thử)
    DRIVE_ROOT = Path(os.environ.get('VNTS_DRIVE_ROOT', './VNTrafficSign'))
    LOCAL_ROOT = Path(os.environ.get('VNTS_LOCAL_ROOT', './local'))

SRC_ZIP = DRIVE_ROOT / 'data' / 'Dataset DAP391m.v4i.yolov8.zip'
OUT_DIR = DRIVE_ROOT / 'output_cleaneddata'          # thư mục kết quả trên Drive
EXTRACT_DIR = LOCAL_ROOT / 'raw_data'                # giải nén cục bộ, không lên Drive
LOCAL_OUT_ZIP = LOCAL_ROOT / 'cleaned_data.zip'      # tạo zip cục bộ trước rồi mới chép lên Drive
OUT_ZIP = OUT_DIR / 'cleaned_data.zip'

NUM_CLASSES_EXPECTED = 115                           # theo data.yaml của bản v4
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

assert SRC_ZIP.exists(), f'Không tìm thấy file zip gốc: {SRC_ZIP}'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print('Zip gốc   :', SRC_ZIP, f'({SRC_ZIP.stat().st_size / 1e6:.1f} MB)')
print('Thư mục ra:', OUT_DIR)

# %% [markdown]
# ## 2. Giải nén vào `/content` (ổ cục bộ của Colab)

# %%
if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True)

with zipfile.ZipFile(SRC_ZIP) as zf:
    bad = zf.testzip()
    assert bad is None, f'File zip gốc bị lỗi ở: {bad}'
    src_members = [m for m in zf.namelist() if not m.endswith('/')]
    zf.extractall(EXTRACT_DIR)
print(f'Đã giải nén {len(src_members)} file vào {EXTRACT_DIR}')

# %% [markdown]
# ## 3. Xác định cấu trúc thư mục và đọc `data.yaml`

# %%
import yaml

yaml_files = list(EXTRACT_DIR.rglob('data.yaml'))
assert len(yaml_files) == 1, f'Cần đúng 1 file data.yaml, tìm thấy {len(yaml_files)}: {yaml_files}'
DATA_ROOT = yaml_files[0].parent
with open(yaml_files[0], encoding='utf-8') as f:
    data_cfg = yaml.safe_load(f)

CLASS_NAMES = list(data_cfg['names'].values()) if isinstance(data_cfg['names'], dict) else list(data_cfg['names'])
NC = int(data_cfg['nc'])
print('Thư mục dữ liệu:', DATA_ROOT.relative_to(EXTRACT_DIR) if DATA_ROOT != EXTRACT_DIR else '(gốc của zip)')
print('nc =', NC, '| số tên class =', len(CLASS_NAMES))
assert NC == len(CLASS_NAMES), 'nc trong data.yaml không khớp số tên class'
if NC != NUM_CLASSES_EXPECTED:
    print(f'⚠️ CẢNH BÁO: nc = {NC}, khác giá trị mong đợi {NUM_CLASSES_EXPECTED}. Hãy kiểm tra lại trước khi dùng.')

# Mỗi split là thư mục chứa cả images/ và labels/
SPLITS = sorted(p.parent.name for p in DATA_ROOT.glob('*/images') if p.is_dir())
print('Các split tìm thấy:', SPLITS)
for s in SPLITS:
    assert (DATA_ROOT / s / 'labels').is_dir(), f'Split {s} thiếu thư mục labels/'

# Các file nằm ngoài split (README, data.yaml...) được giữ nguyên
others = [p.relative_to(DATA_ROOT) for p in DATA_ROOT.iterdir() if p.is_file()]
print('File khác ở gốc dữ liệu:', [str(o) for o in others])

# %% [markdown]
# ## 4. Quét toàn bộ ảnh và nhãn
#
# Với mỗi ảnh, file nhãn tương ứng là `labels/<tên ảnh không đuôi>.txt`.

# %%
def check_label_lines(text):
    """Trả về (số box hợp lệ, danh sách vấn đề) của một file nhãn. Chỉ để báo cáo."""
    n_ok, issues = 0, []
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append((ln, f'khong_dung_5_gia_tri ({len(parts)} gia tri)', line.strip()[:120]))
            continue
        try:
            cid = float(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            issues.append((ln, 'gia_tri_khong_phai_so', line.strip()[:120]))
            continue
        if cid != int(cid) or not (0 <= int(cid) < NC):
            issues.append((ln, f'class_id_ngoai_pham_vi ({parts[0]})', line.strip()[:120]))
            continue
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            issues.append((ln, 'toa_do_ngoai_[0,1]', line.strip()[:120]))
        if w <= 0 or h <= 0:
            issues.append((ln, 'box_kich_thuoc_bang_0', line.strip()[:120]))
        n_ok += 1
    return n_ok, issues


img_rows, label_issue_rows, orphan_rows, non_image_rows = [], [], [], []

for split in SPLITS:
    img_dir, lbl_dir = DATA_ROOT / split / 'images', DATA_ROOT / split / 'labels'
    images = {}
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images[p.stem] = p
        elif p.is_file():
            non_image_rows.append({'split': split, 'file': str(p.relative_to(EXTRACT_DIR))})
    labels = {p.stem: p for p in lbl_dir.glob('*.txt')}

    for stem, ip in images.items():
        lp = labels.get(stem)
        row = {'split': split, 'image': str(ip.relative_to(EXTRACT_DIR)),
               'label': str(lp.relative_to(EXTRACT_DIR)) if lp else '',
               'n_lines': 0, 'n_valid_boxes': 0, 'status': ''}
        if lp is None:
            row['status'] = 'khong_co_file_nhan'
        else:
            text = lp.read_text(encoding='utf-8', errors='replace')
            lines = [l for l in text.splitlines() if l.strip()]
            row['n_lines'] = len(lines)
            if not lines:
                row['status'] = 'file_nhan_rong'
            else:
                row['status'] = 'co_nhan'
                n_ok, issues = check_label_lines(text)
                row['n_valid_boxes'] = n_ok
                for ln, issue, content in issues:
                    label_issue_rows.append({'split': split, 'label': row['label'], 'line': ln,
                                             'issue': issue, 'content': content})
        # Kiểm tra ảnh hỏng (chỉ báo cáo)
        try:
            with Image.open(ip) as im:
                im.verify()
            row['image_ok'] = True
        except Exception as e:
            row['image_ok'] = False
            row['image_error'] = str(e)[:200]
        img_rows.append(row)

    for stem, lp in labels.items():
        if stem not in images:
            orphan_rows.append({'split': split, 'label': str(lp.relative_to(EXTRACT_DIR))})

df = pd.DataFrame(img_rows)
if 'image_error' not in df:
    df['image_error'] = ''
df['image_error'] = df['image_error'].fillna('')
print('Tổng số ảnh quét được:', len(df))
display(pd.crosstab(df['split'], df['status'], margins=True, margins_name='Tổng'))

# %% [markdown]
# ## 5. Xác định ảnh bị loại (chỉ ảnh không có nhãn)

# %%
REMOVE_STATUSES = {'khong_co_file_nhan', 'file_nhan_rong'}
df['removed'] = df['status'].isin(REMOVE_STATUSES)
removed = df[df['removed']].copy()

reason_map = {'khong_co_file_nhan': 'Không có file nhãn', 'file_nhan_rong': 'File nhãn rỗng'}
removed['reason'] = removed['status'].map(reason_map)
removed[['split', 'image', 'label', 'reason']].to_csv(OUT_DIR / 'removed_images.csv', index=False, encoding='utf-8-sig')
print(f'Số ảnh bị loại: {len(removed)} / {len(df)} ({len(removed) / max(len(df), 1):.2%})')
display(removed.groupby(['split', 'reason']).size().rename('số ảnh').reset_index())

# %% [markdown]
# ## 6. Các vấn đề chỉ báo cáo (KHÔNG loại)
#
# Những ảnh dưới đây vẫn nằm trong `cleaned_data.zip`. Nhóm cần tự soát lại nếu muốn xử lý.

# %%
kept = df[~df['removed']]

corrupted = kept[~kept['image_ok']][['split', 'image', 'image_error']]
corrupted.to_csv(OUT_DIR / 'report_corrupted_images.csv', index=False, encoding='utf-8-sig')

label_issues = pd.DataFrame(label_issue_rows, columns=['split', 'label', 'line', 'issue', 'content'])
label_issues.to_csv(OUT_DIR / 'report_label_issues.csv', index=False, encoding='utf-8-sig')

orphans = pd.DataFrame(orphan_rows, columns=['split', 'label'])
orphans.to_csv(OUT_DIR / 'report_orphan_labels.csv', index=False, encoding='utf-8-sig')

non_images = pd.DataFrame(non_image_rows, columns=['split', 'file'])
non_images.to_csv(OUT_DIR / 'report_non_image_files.csv', index=False, encoding='utf-8-sig')

print('Ảnh hỏng (vẫn giữ)                        :', len(corrupted))
print('Dòng nhãn có vấn đề (vẫn giữ)              :', len(label_issues),
      f'trong {label_issues["label"].nunique()} file nhãn')
print('File nhãn không có ảnh tương ứng (vẫn giữ):', len(orphans))
print('File không phải ảnh trong thư mục images/  :', len(non_images))
if len(label_issues):
    issue_type = label_issues['issue'].str.replace(r' \(.*\)$', '', regex=True)
    display(issue_type.value_counts().rename('số dòng').to_frame())
    display(label_issues.head(10))
if len(corrupted):
    display(corrupted.head(10))

# %% [markdown]
# ## 7. Tạo `cleaned_data.zip` (cùng cấu trúc với zip gốc)
#
# Zip mới chứa mọi file của zip gốc, trừ các ảnh bị loại và file nhãn rỗng đi kèm. Đường dẫn bên trong zip giữ nguyên.
# Ảnh đã được nén sẵn nên được lưu ở chế độ `STORED` (không nén lại) cho nhanh; file văn bản được nén `DEFLATED`.

# %%
drop_members = set(removed['image']) | set(removed.loc[removed['label'] != '', 'label'])
drop_members = {m.replace(os.sep, '/') for m in drop_members}

if LOCAL_OUT_ZIP.exists():
    LOCAL_OUT_ZIP.unlink()

written = 0
with zipfile.ZipFile(LOCAL_OUT_ZIP, 'w') as zout:
    for member in src_members:
        if member in drop_members:
            continue
        src = EXTRACT_DIR / member
        ctype = zipfile.ZIP_STORED if src.suffix.lower() in IMG_EXTS else zipfile.ZIP_DEFLATED
        zout.write(src, arcname=member, compress_type=ctype)
        written += 1

assert written == len(src_members) - len(drop_members), 'Số file ghi vào zip không khớp'
print(f'Đã ghi {written} file (gốc {len(src_members)}, bỏ {len(drop_members)})')
print(f'Kích thước: {LOCAL_OUT_ZIP.stat().st_size / 1e6:.1f} MB')

# %% [markdown]
# ## 8. Kiểm tra lại `cleaned_data.zip`
#
# Mở lại zip vừa tạo và đếm độc lập: mọi ảnh phải có file nhãn không rỗng, và số ảnh phải khớp với bước 5.

# %%
with zipfile.ZipFile(LOCAL_OUT_ZIP) as zf:
    assert zf.testzip() is None, 'cleaned_data.zip bị lỗi'
    names = [n for n in zf.namelist() if not n.endswith('/')]
    name_set = set(names)
    check_rows = []
    for n in names:
        p = Path(n)
        if p.parent.name != 'images' or p.suffix.lower() not in IMG_EXTS:
            continue
        split = p.parent.parent.name
        lbl = str(p.parent.parent / 'labels' / (p.stem + '.txt')).replace(os.sep, '/')
        has_label = lbl in name_set and zf.read(lbl).decode('utf-8', 'replace').strip() != ''
        check_rows.append({'split': split, 'has_nonempty_label': has_label})

chk = pd.DataFrame(check_rows)
assert chk['has_nonempty_label'].all(), 'Vẫn còn ảnh không có nhãn trong cleaned_data.zip!'

before = df.groupby('split').size()
after_expected = kept.groupby('split').size().reindex(before.index, fill_value=0)
after_actual = chk.groupby('split').size().reindex(before.index, fill_value=0)
assert (after_expected == after_actual).all(), 'Số ảnh trong zip không khớp kết quả bước 5'

summary = pd.DataFrame({
    'ảnh trước khi clean': before,
    'loại: không có file nhãn': df[df['status'] == 'khong_co_file_nhan'].groupby('split').size().reindex(before.index, fill_value=0),
    'loại: file nhãn rỗng': df[df['status'] == 'file_nhan_rong'].groupby('split').size().reindex(before.index, fill_value=0),
    'ảnh sau khi clean': after_actual,
    'số dòng nhãn (box) sau khi clean': kept.groupby('split')['n_lines'].sum().reindex(before.index, fill_value=0),
})
summary.loc['Tổng'] = summary.sum()
summary.index.name = 'split'
summary.to_csv(OUT_DIR / 'clean_summary.csv', encoding='utf-8-sig')
print('✅ Kiểm tra đạt: mọi ảnh trong cleaned_data.zip đều có nhãn.')
display(summary)

# %% [markdown]
# ## 9. Chép `cleaned_data.zip` lên Drive

# %%
shutil.copy2(LOCAL_OUT_ZIP, OUT_ZIP)
assert OUT_ZIP.stat().st_size == LOCAL_OUT_ZIP.stat().st_size
print('Đã lưu:', OUT_ZIP)
print('Các file trong', OUT_DIR.name + ':')
for p in sorted(OUT_DIR.iterdir()):
    print(f'  {p.name:35s} {p.stat().st_size / 1e6:10.2f} MB')
