# %% [markdown]
# # 03 – Gom nhóm ảnh gần trùng (dHash) và chia train : valid : test = 70 : 15 : 15
#
# **Đầu vào:** `MyDrive/VNTrafficSign/output_cleaneddata/cleaned_data.zip` (kết quả của notebook 01).
#
# **Đầu ra** (thư mục `MyDrive/VNTrafficSign/output_split/`):
# - `split_data_by_box.zip`: chia có phân tầng theo **số box** mỗi class
# - `split_data_by_image.zip`: chia có phân tầng theo **số ảnh chứa** mỗi class
# - `tables/`, `figures/`: bảng số liệu và hình để so sánh hai cách chia, ảnh mẫu để xem tay ngưỡng dHash
#
# Cả hai bộ đều **chưa tăng cường dữ liệu**. Nhóm xem số liệu thực tế rồi chọn một bộ.
#
# **Quy trình (đã chốt với nhóm ngày 16/10/2026)**
#
# | Bước | Việc | Căn cứ |
# |---|---|---|
# | 1 | Gộp ảnh của train/valid/test cũ thành một tập | Dữ liệu được chia lại từ đầu |
# | 2 | Dòng nhãn polygon → tự đổi thành bbox (min/max tọa độ) | Nhóm chốt |
# | 3 | Dòng có class id ngoài 0–114: nếu < 10 ảnh bị ảnh hưởng thì **loại các ảnh đó**; nếu ≥ 10 ảnh thì **dừng lại** để nhóm kiểm tra | Nhóm chốt |
# | 4 | Ảnh hỏng, dòng nhãn sai định dạng khác: **dừng lại** và xuất danh sách (nhóm chưa chốt cách xử lý) | Không tự quyết |
# | 5 | Tính **dHash** 64 bit cho mọi ảnh; hai ảnh có khoảng cách Hamming ≤ `DHASH_MAX_DIST` được nối với nhau | Zauner 2010; Vallez và cộng sự 2022 |
# | 6 | Gom các cặp thành nhóm (thành phần liên thông, union-find) | Nhóm = mọi ảnh nối được với nhau qua chuỗi cặp gần trùng |
# | 7 | In danh sách class chỉ có 1–2 nhóm ảnh | Để nhóm quyết cách xử lý |
# | 8 | Chia **từng nhóm** (không chia lẻ ảnh) bằng iterative stratification ở cấp nhóm | Sechidis và cộng sự 2011 (ECML PKDD) |
# | 9 | Kiểm tra rò rỉ: không cặp ảnh gần trùng nào nằm ở hai phần khác nhau | D-LeDe, Babu và cộng sự 2025 (VEHITS) |
#
# Không dùng tên file gốc để gom nhóm (nhóm chốt).

# %% [markdown]
# ## Cấu hình

# %%
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
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
OUT_DIR = DRIVE_ROOT / 'output_split'
EXTRACT_DIR = LOCAL_ROOT / 'split_work' / 'cleaned'
MERGED_DIR = LOCAL_ROOT / 'split_work' / 'merged'
BUILD_DIR = LOCAL_ROOT / 'split_work' / 'build'
FIG_DIR, TAB_DIR = OUT_DIR / 'figures', OUT_DIR / 'tables'
for d in (OUT_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Tham số ----
SEED = 42
RATIOS = {'train': 0.70, 'valid': 0.15, 'test': 0.15}
DHASH_MAX_DIST = 2          # ngưỡng gom nhóm (Hamming trên 64 bit). Nhóm chốt = 2 sau khi xem tay (17/10/2026)
REVIEW_MAX_DIST = 12        # tìm cặp tới khoảng cách này để xem tay và kiểm tra rò rỉ mở rộng
OUT_OF_RANGE_MAX_IMAGES = 10  # < 10 ảnh có class id ngoài phạm vi → loại; ≥ 10 → dừng
RARE_MAX_GROUPS = 2         # class có ≤ 2 nhóm ảnh được liệt kê
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

random.seed(SEED)
np.random.seed(SEED)
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})
SPLIT_COLOR = {'train': '#2a78d6', 'valid': '#eb6834', 'test': '#1baf7a'}


def savetab(df, name, index=False):
    df.to_csv(TAB_DIR / f'{name}.csv', index=index, encoding='utf-8-sig')


def savefig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.png')
    plt.show()


assert CLEAN_ZIP.exists(), f'Không tìm thấy {CLEAN_ZIP}. Hãy chạy notebook 01_clean_data trước.'
print('Đầu vào:', CLEAN_ZIP)
print('Đầu ra :', OUT_DIR)

# %% [markdown]
# ## 1. Giải nén vào `/content` và gộp thành một tập

# %%
for d in (EXTRACT_DIR, MERGED_DIR, BUILD_DIR):
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
OLD_SPLITS = sorted(p.parent.name for p in DATA_ROOT.glob('*/images') if p.is_dir())
print('nc =', NC, '| thư mục cũ:', OLD_SPLITS)

stem_seen = defaultdict(list)
for s in OLD_SPLITS:
    for p in (DATA_ROOT / s / 'images').iterdir():
        if p.suffix.lower() in IMG_EXTS:
            stem_seen[p.stem].append(s)
clash = {k: v for k, v in stem_seen.items() if len(v) > 1}
if clash:
    savetab(pd.DataFrame([(k, ','.join(v)) for k, v in clash.items()], columns=['stem', 'splits']), 'merge_name_clashes')
    raise RuntimeError(f'Có {len(clash)} tên file trùng giữa các split cũ (tables/merge_name_clashes.csv). Cần nhóm quyết định.')

(MERGED_DIR / 'images').mkdir(parents=True)
(MERGED_DIR / 'labels').mkdir(parents=True)
rows = []
for s in OLD_SPLITS:
    for p in sorted((DATA_ROOT / s / 'images').iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        shutil.move(str(p), MERGED_DIR / 'images' / p.name)
        lp = DATA_ROOT / s / 'labels' / f'{p.stem}.txt'
        if lp.exists():
            shutil.move(str(lp), MERGED_DIR / 'labels' / lp.name)
        rows.append({'image': p.name, 'stem': p.stem})
    shutil.rmtree(DATA_ROOT / s)   # không giữ lại thông tin split cũ
images = pd.DataFrame(rows)
print(f'✅ Đã gộp {len(images)} ảnh thành một tập (không còn train/valid/test cũ)')

# %% [markdown]
# ## 2. Đọc nhãn: polygon → bbox, kiểm tra class id, ảnh hỏng
#
# - **Polygon** (`class x1 y1 x2 y2 … xn yn`, từ 3 điểm trở lên): đổi thành bbox bằng min/max của tọa độ, cắt về [0, 1].
# - **Class id ngoài 0–114:** đếm số ảnh có ít nhất một dòng như vậy. Dưới `OUT_OF_RANGE_MAX_IMAGES` ảnh → loại các ảnh đó khỏi bộ chia. Từ `OUT_OF_RANGE_MAX_IMAGES` ảnh trở lên → dừng, xuất danh sách để nhóm kiểm tra.
# - **Ảnh hỏng** hoặc **dòng sai định dạng khác** (số giá trị lẻ, giá trị không phải số…): dừng, xuất danh sách. Nhóm chưa chốt cách xử lý nên notebook không tự loại.

# %%
def parse_label_file(path):
    """Trả về (danh_sách_dòng_mới, danh_sách_box, các vấn đề). Box = (class_id, xc, yc, w, h)."""
    out_lines, boxes, issues, n_poly = [], [], [], 0
    if not path.exists():
        return out_lines, boxes, [('khong_co_file_nhan', 0, '')], 0
    for ln, raw in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            issues.append(('gia_tri_khong_phai_so', ln, line[:120]))
            continue
        cid = vals[0]
        if cid != int(cid):
            issues.append(('class_id_khong_nguyen', ln, line[:120]))
            continue
        cid = int(cid)
        coords = vals[1:]
        if len(coords) == 4:
            xc, yc, w, h = coords
            new_line = line
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = np.clip(coords[0::2], 0, 1)
            ys = np.clip(coords[1::2], 0, 1)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            new_line = f'{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}'
            n_poly += 1
        else:
            issues.append((f'so_gia_tri_khong_hop_le ({len(parts)})', ln, line[:120]))
            continue
        if not (0 <= cid < NC):
            issues.append(('class_id_ngoai_pham_vi', ln, line[:120]))
            continue
        out_lines.append(new_line)
        boxes.append((cid, xc, yc, w, h))
    return out_lines, boxes, issues, n_poly


def image_ok(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


label_lines, label_boxes, issue_rows = {}, {}, []
n_poly_total, broken = 0, []
for name, stem in zip(images['image'], images['stem']):
    if not image_ok(MERGED_DIR / 'images' / name):
        broken.append(name)
    lines, bxs, issues, n_poly = parse_label_file(MERGED_DIR / 'labels' / f'{stem}.txt')
    label_lines[name], label_boxes[name] = lines, bxs
    n_poly_total += n_poly
    for kind, ln, content in issues:
        issue_rows.append({'image': name, 'line': ln, 'issue': kind, 'content': content})

issues_df = pd.DataFrame(issue_rows, columns=['image', 'line', 'issue', 'content'])
savetab(issues_df, 'label_issues_all')
savetab(pd.DataFrame({'image': broken}), 'broken_images')

oor_images = sorted(issues_df.loc[issues_df['issue'] == 'class_id_ngoai_pham_vi', 'image'].unique())
other_issues = issues_df[issues_df['issue'] != 'class_id_ngoai_pham_vi']
print(f'Dòng polygon đã đổi thành bbox  : {n_poly_total}')
print(f'Ảnh có class id ngoài 0–{NC - 1}     : {len(oor_images)}')
print(f'Dòng sai định dạng khác          : {len(other_issues)} (trong {other_issues["image"].nunique()} ảnh)')
print(f'Ảnh hỏng                         : {len(broken)}')

stop_msgs = []
if broken:
    stop_msgs.append(f'{len(broken)} ảnh hỏng (tables/broken_images.csv)')
if len(other_issues):
    stop_msgs.append(f'{len(other_issues)} dòng nhãn sai định dạng (tables/label_issues_all.csv)')
if len(oor_images) >= OUT_OF_RANGE_MAX_IMAGES:
    savetab(issues_df[issues_df['issue'] == 'class_id_ngoai_pham_vi'], 'class_id_out_of_range_TO_CHECK')
    stop_msgs.append(f'{len(oor_images)} ảnh có class id ngoài phạm vi (≥ {OUT_OF_RANGE_MAX_IMAGES}, '
                     'tables/class_id_out_of_range_TO_CHECK.csv)')
if stop_msgs:
    raise RuntimeError('DỪNG để nhóm kiểm tra: ' + '; '.join(stop_msgs))

removed = pd.DataFrame({'image': oor_images, 'reason': f'class id ngoài 0–{NC - 1} (< {OUT_OF_RANGE_MAX_IMAGES} ảnh → loại)'})
savetab(removed, 'removed_images')
images = images[~images['image'].isin(oor_images)].reset_index(drop=True)
print(f'Đã loại {len(removed)} ảnh có class id ngoài phạm vi. Còn {len(images)} ảnh.')
no_box = [n for n in images['image'] if not label_boxes[n]]
if no_box:
    print(f'⚠️ {len(no_box)} ảnh không còn box hợp lệ nào (vẫn giữ, xem tables/images_without_boxes.csv)')
    savetab(pd.DataFrame({'image': no_box}), 'images_without_boxes')

# %% [markdown]
# ## 3. Tính dHash và tìm các cặp ảnh gần trùng
#
# dHash: thu ảnh xám về 9×8, so sánh từng cặp pixel liền kề theo hàng ngang → 64 bit. Khoảng cách giữa hai ảnh là số bit khác nhau (Hamming, 0–64).
# Ảnh được xoay theo cờ EXIF trước khi tính, giống cách YOLO đọc ảnh khi train.

# %%
def dhash(path, size=8):
    with Image.open(path) as im:
        g = np.asarray(ImageOps.exif_transpose(im).convert('L').resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return np.uint64(int(''.join('1' if b else '0' for b in bits), 2))


names = images['image'].tolist()
N = len(names)
hashes = np.array([dhash(MERGED_DIR / 'images' / n) for n in names], dtype=np.uint64)
images['dhash'] = [f'{int(h):016x}' for h in hashes]

if hasattr(np, 'bitwise_count'):
    popcount = np.bitwise_count
else:
    _lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    popcount = lambda x: _lut[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)

pi, pj, pd_ = [], [], []
CH = 512
for s in range(0, N, CH):
    d = popcount(hashes[s:s + CH, None] ^ hashes[None, :]).astype(np.int16)
    ii, jj = np.nonzero(d <= REVIEW_MAX_DIST)
    keep = jj > ii + s
    pi.append(ii[keep] + s)
    pj.append(jj[keep])
    pd_.append(d[ii[keep], jj[keep]])
pairs = pd.DataFrame({'i': np.concatenate(pi) if pi else [], 'j': np.concatenate(pj) if pj else [],
                      'dist': np.concatenate(pd_) if pd_ else []}).astype(int)
pairs['image_a'] = [names[i] for i in pairs['i']]
pairs['image_b'] = [names[j] for j in pairs['j']]
savetab(pairs[['image_a', 'image_b', 'dist']], 'dhash_pairs_upto_review_max')
dist_hist = pairs['dist'].value_counts().reindex(range(REVIEW_MAX_DIST + 1), fill_value=0)
print(f'Số cặp có Hamming ≤ {REVIEW_MAX_DIST}: {len(pairs)}')
display(dist_hist.rename('số cặp').rename_axis('Hamming').to_frame().T)

# %% [markdown]
# ## 4. Xem tay để chốt ngưỡng `DHASH_MAX_DIST`
#
# Mỗi hình dưới đây là các cặp ảnh ngẫu nhiên ở một khoảng cách Hamming. 3 thành viên xem và trả lời: ở khoảng cách nào hai ảnh **vẫn là cùng một cảnh** (cùng biển, cùng vị trí chụp)?
# Ngưỡng nên đặt ở khoảng cách lớn nhất mà phần lớn cặp vẫn cùng cảnh. Sửa `DHASH_MAX_DIST` ở ô cấu hình rồi chạy lại từ mục 5.
# Hình được lưu ở `output_split/figures/04_review_dist_XX.png`.

# %%
def thumb(name, side=320):
    with Image.open(MERGED_DIR / 'images' / name) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((side, side))
        return im.copy()


N_REVIEW = 6
rng_review = random.Random(SEED)
for dist in range(REVIEW_MAX_DIST + 1):
    sub = pairs[pairs['dist'] == dist]
    if sub.empty:
        continue
    pick = sub.iloc[rng_review.sample(range(len(sub)), min(N_REVIEW, len(sub)))]
    fig, axes = plt.subplots(len(pick), 2, figsize=(7, 2.3 * len(pick)))
    axes = np.atleast_2d(axes)
    for row, p in zip(axes, pick.itertuples()):
        for ax, n in zip(row, (p.image_a, p.image_b)):
            ax.imshow(thumb(n))
            ax.set_title(n[:45], fontsize=7)
            ax.axis('off')
    fig.suptitle(f'Hamming = {dist}  ({len(sub)} cặp, hiển thị {len(pick)})', fontweight='bold')
    savefig(fig, f'04_review_dist_{dist:02d}')

# %% [markdown]
# ## 5. Gom nhóm (union-find) với ngưỡng đã chốt

# %%
parent = list(range(N))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


linked = pairs[pairs['dist'] <= DHASH_MAX_DIST]
for i, j in zip(linked['i'], linked['j']):
    ri, rj = find(i), find(j)
    if ri != rj:
        parent[ri] = rj

roots = [find(i) for i in range(N)]
root_to_gid = {}
for r in roots:                      # đánh số nhóm theo thứ tự xuất hiện
    root_to_gid.setdefault(r, len(root_to_gid))
images['group_id'] = [root_to_gid[r] for r in roots]
gsize = images['group_id'].value_counts()
images['group_size'] = images['group_id'].map(gsize)

group_summary = pd.DataFrame([
    ('Ngưỡng dHash (Hamming ≤)', DHASH_MAX_DIST),
    ('Số ảnh', N),
    ('Số cặp được nối', len(linked)),
    ('Số nhóm', int(gsize.size)),
    ('Nhóm chỉ có 1 ảnh', int((gsize == 1).sum())),
    ('Nhóm có ≥ 2 ảnh', int((gsize >= 2).sum())),
    ('Số ảnh nằm trong nhóm ≥ 2 ảnh', int(gsize[gsize >= 2].sum())),
    ('Nhóm lớn nhất (số ảnh)', int(gsize.max())),
], columns=['chỉ số', 'giá trị'])
display(group_summary)
savetab(group_summary, 'group_summary')
savetab(gsize.value_counts().sort_index().rename_axis('group_size').rename('n_groups').reset_index(), 'group_size_hist')
if gsize.max() > min(RATIOS.values()) * N:
    print(f'⚠️ Nhóm lớn nhất có {gsize.max()} ảnh, lớn hơn 15% tổng số ảnh: nhóm này chỉ có thể nằm ở train. '
          'Có thể ngưỡng đang quá rộng (chuỗi cặp nối dài). Xem hình 05_largest_groups.')

fig, ax = plt.subplots(figsize=(8, 3.5))
vc = gsize.value_counts().sort_index()
ax.bar(vc.index.astype(str), vc.values, color='#2a78d6')
ax.set_yscale('log')
ax.set_xlabel('Số ảnh trong nhóm')
ax.set_ylabel('Số nhóm (log)')
ax.set_title(f'Kích thước nhóm ảnh gần trùng (dHash ≤ {DHASH_MAX_DIST})')
savefig(fig, '05_group_size_hist')

top_groups = gsize[gsize >= 2].head(5)
if len(top_groups):
    fig, axes = plt.subplots(len(top_groups), 6, figsize=(14, 2.2 * len(top_groups)))
    axes = np.atleast_2d(axes)
    for row, (gid, sz) in zip(axes, top_groups.items()):
        members = images.loc[images['group_id'] == gid, 'image'].tolist()[:6]
        for ax in row:
            ax.axis('off')
        for ax, n in zip(row, members):
            ax.imshow(thumb(n, 240))
            ax.set_title(n[:28], fontsize=6)
        row[0].text(-0.05, 0.5, f'nhóm {gid}\n{sz} ảnh', transform=row[0].transAxes, ha='right', va='center', fontsize=9)
    fig.suptitle('5 nhóm lớn nhất (tối đa 6 ảnh mỗi nhóm)', fontweight='bold')
    savefig(fig, '05_largest_groups')

# %% [markdown]
# ## 6. Bảng box và danh sách class chỉ có 1–2 nhóm ảnh

# %%
box_rows = [{'image': n, 'class_id': c} for n in names for (c, *_r) in label_boxes[n]]
boxes = pd.DataFrame(box_rows).merge(images[['image', 'group_id']], on='image')
boxes['class_name'] = boxes['class_id'].map(dict(enumerate(CLASS_NAMES)))

per_class = pd.DataFrame({'class_id': range(NC), 'class_name': CLASS_NAMES})
per_class['n_groups'] = per_class['class_id'].map(boxes.groupby('class_id')['group_id'].nunique()).fillna(0).astype(int)
per_class['n_images'] = per_class['class_id'].map(boxes.groupby('class_id')['image'].nunique()).fillna(0).astype(int)
per_class['n_boxes'] = per_class['class_id'].map(boxes['class_id'].value_counts()).fillna(0).astype(int)
savetab(per_class, 'class_group_counts')

rare = per_class[per_class['n_groups'] <= RARE_MAX_GROUPS].sort_values(['n_groups', 'n_boxes'])
savetab(rare, 'classes_with_le2_groups')
print(f'Có {len(rare)} class chỉ có ≤ {RARE_MAX_GROUPS} nhóm ảnh (không thể có mặt ở cả 3 phần):')
display(rare[['class_id', 'class_name', 'n_groups', 'n_images', 'n_boxes']])
print('→ Nhóm xem danh sách này rồi quyết cách xử lý. Bên dưới notebook vẫn chia bình thường để nhóm thấy số liệu thực tế; '
      'mục 9 cho biết các class này đang rơi vào phần nào.')

# %% [markdown]
# ## 7. Chia theo nhóm bằng iterative stratification
#
# Thuật toán của Sechidis và cộng sự (2011), áp dụng ở **cấp nhóm**:
# 1. Mỗi phần có "chỉ tiêu" cho từng class = tỉ lệ (70/15/15%) × tổng số đơn vị của class đó.
# 2. Lặp: chọn class **hiếm nhất** trong số nhóm chưa chia; lần lượt đưa từng nhóm chứa class đó (nhóm lớn trước) vào phần còn thiếu class đó nhiều nhất. Hòa thì chọn phần còn thiếu nhiều đơn vị tổng nhất, hòa nữa thì chọn ngẫu nhiên theo `SEED`.
# 3. Trừ chỉ tiêu của mọi class có trong nhóm vừa chia.
#
# Chạy 2 lần với 2 đơn vị:
# - **box**: chỉ tiêu tính theo số box của từng class; tổng thể tính theo tổng số box.
# - **image**: chỉ tiêu tính theo số ảnh chứa từng class; tổng thể tính theo số ảnh.

# %%
SPLITS = list(RATIOS)
R = np.array([RATIOS[s] for s in SPLITS])
gids = np.sort(images['group_id'].unique())
gindex = {g: k for k, g in enumerate(gids)}
G = len(gids)

M_box = np.zeros((G, NC), dtype=np.int64)
for g, c in zip(boxes['group_id'], boxes['class_id']):
    M_box[gindex[g], c] += 1
img_cls = boxes[['image', 'group_id', 'class_id']].drop_duplicates()
M_img = np.zeros((G, NC), dtype=np.int64)
for g, c in zip(img_cls['group_id'], img_cls['class_id']):
    M_img[gindex[g], c] += 1
size_box = M_box.sum(axis=1)
size_img = images.groupby('group_id').size().reindex(gids).values


def iterative_group_split(M, size, ratios, seed):
    rng = np.random.default_rng(seed)
    G_, C_ = M.shape
    desired = np.outer(ratios, M.sum(0)).astype(float)
    desired_total = ratios * size.sum()
    assign = np.full(G_, -1)
    unassigned = np.ones(G_, dtype=bool)

    def pick_split(scores_primary):
        best = np.flatnonzero(np.isclose(scores_primary, scores_primary.max()))
        if len(best) > 1:
            sec = desired_total[best]
            best = best[np.isclose(sec, sec.max())]
        return int(rng.choice(best)) if len(best) > 1 else int(best[0])

    while unassigned.any():
        rem = M[unassigned].sum(0)
        avail = np.flatnonzero(rem > 0)
        if len(avail) == 0:   # nhóm không còn class nào (ảnh không có box hợp lệ)
            for g in np.flatnonzero(unassigned)[np.argsort(-size[unassigned], kind='stable')]:
                s = pick_split(desired_total.copy())
                assign[g] = s
                desired_total[s] -= size[g]
                unassigned[g] = False
            break
        lmin = rem[avail].min()
        cands = avail[rem[avail] == lmin]
        lab = int(rng.choice(cands)) if len(cands) > 1 else int(cands[0])
        gs = np.flatnonzero(unassigned & (M[:, lab] > 0))
        gs = gs[np.argsort(-M[gs, lab], kind='stable')]
        for g in gs:
            s = pick_split(desired[:, lab].copy())
            assign[g] = s
            desired[s] -= M[g]
            desired_total[s] -= size[g]
            unassigned[g] = False
    return assign


results = {}
for mode, M, size in (('box', M_box, size_box), ('image', M_img, size_img)):
    a = iterative_group_split(M, size, R, SEED)
    split_of_group = {g: SPLITS[a[gindex[g]]] for g in gids}
    results[mode] = images['group_id'].map(split_of_group).values
    images[f'split_by_{mode}'] = results[mode]
print('Đã chia xong 2 phương án: split_by_box, split_by_image')

# %% [markdown]
# ## 8. Kiểm tra rò rỉ
#
# Bắt buộc: **không có cặp nào với Hamming ≤ `DHASH_MAX_DIST` nằm ở hai phần khác nhau** (đúng theo cách xây dựng nhóm).

# %%
leak_rows = []
for mode in results:
    sp = images[f'split_by_{mode}'].values
    cross = sp[pairs['i'].values] != sp[pairs['j'].values]
    strict = int((cross & (pairs['dist'].values <= DHASH_MAX_DIST)).sum())
    assert strict == 0, f'Rò rỉ ở phương án {mode}: {strict} cặp gần trùng nằm ở 2 phần khác nhau'
    grp_split = images.groupby('group_id')[f'split_by_{mode}'].nunique()
    assert (grp_split == 1).all(), f'Có nhóm bị tách ở phương án {mode}'
    leak_rows.append({'phương án': f'theo {mode}', f'cặp Hamming ≤ {DHASH_MAX_DIST} khác phần (phải = 0)': strict})
leak = pd.DataFrame(leak_rows)
display(leak)
savetab(leak, 'leak_check')
print('✅ Không nhóm nào bị tách, không cặp gần trùng nào nằm ở hai phần khác nhau.')

# %% [markdown]
# ## 9. So sánh số liệu thực tế của hai phương án

# %%
def split_report(mode):
    col = f'split_by_{mode}'
    img_split = images.set_index('image')[col]
    b = boxes.assign(split=boxes['image'].map(img_split))
    overall = pd.DataFrame({
        'số ảnh': images[col].value_counts().reindex(SPLITS),
        'số box': b['split'].value_counts().reindex(SPLITS),
        'số nhóm': images.groupby(col)['group_id'].nunique().reindex(SPLITS),
    }).fillna(0).astype(int)
    overall['% ảnh'] = (overall['số ảnh'] / overall['số ảnh'].sum() * 100).round(2)
    overall['% box'] = (overall['số box'] / overall['số box'].sum() * 100).round(2)
    overall['mục tiêu %'] = [RATIOS[s] * 100 for s in SPLITS]
    cls_box = pd.crosstab(b['class_name'], b['split']).reindex(columns=SPLITS, fill_value=0)
    cls_img = pd.crosstab(b.drop_duplicates(['image', 'class_id'])['class_name'],
                          b.drop_duplicates(['image', 'class_id'])['split']).reindex(columns=SPLITS, fill_value=0)
    cls_box = cls_box.reindex(CLASS_NAMES, fill_value=0)
    cls_img = cls_img.reindex(CLASS_NAMES, fill_value=0)
    return overall, cls_box, cls_img


def class_deviation(cls_tab):
    tot = cls_tab.sum(axis=1).replace(0, np.nan)
    share = cls_tab.div(tot, axis=0)
    return (share - pd.Series(RATIOS)).abs().mean(axis=1)   # độ lệch tuyệt đối trung bình so với 70/15/15


compare_rows, reports = [], {}
for mode in results:
    overall, cls_box, cls_img = split_report(mode)
    reports[mode] = (overall, cls_box, cls_img)
    print(f'\n=== Phương án phân tầng theo {mode.upper()} ===')
    display(overall)
    savetab(overall, f'split_overall_by_{mode}', index=True)
    savetab(cls_box.add_prefix('box_').join(cls_img.add_prefix('img_')), f'split_per_class_by_{mode}', index=True)
    present = cls_box.sum(axis=1) > 0
    compare_rows.append({
        'phương án': f'theo {mode}',
        '% ảnh train/valid/test': ' / '.join(f'{v:.1f}' for v in overall['% ảnh']),
        '% box train/valid/test': ' / '.join(f'{v:.1f}' for v in overall['% box']),
        'lệch TB theo box mỗi class (điểm %)': round(class_deviation(cls_box[present]).mean() * 100, 2),
        'lệch TB theo ảnh mỗi class (điểm %)': round(class_deviation(cls_img[present]).mean() * 100, 2),
        'class có box nhưng vắng ở valid': int(((cls_box['valid'] == 0) & present).sum()),
        'class có box nhưng vắng ở test': int(((cls_box['test'] == 0) & present).sum()),
    })
compare = pd.DataFrame(compare_rows)
print('\n=== So sánh hai phương án ===')
display(compare)
savetab(compare, 'compare_box_vs_image')

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
for ax, mode in zip(axes, results):
    overall = reports[mode][0]
    x = np.arange(2)
    left = np.zeros(2)
    for s in SPLITS:
        vals = np.array([overall.loc[s, '% ảnh'], overall.loc[s, '% box']])
        ax.barh(x, vals, left=left, color=SPLIT_COLOR[s], label=s, edgecolor='white', linewidth=2)
        for xi, (v, l0) in enumerate(zip(vals, left)):
            ax.text(l0 + v / 2, xi, f'{v:.1f}%', ha='center', va='center', color='white', fontsize=9)
        left += vals
    ax.set_yticks(x, ['% ảnh', '% box'])
    ax.axvline(70, color='#52514e', ls='--', lw=0.8)
    ax.axvline(85, color='#52514e', ls='--', lw=0.8)
    ax.set_xlim(0, 100)
    ax.set_title(f'Phân tầng theo {mode}')
axes[0].legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(1.05, -0.15))
fig.suptitle('Tỉ lệ thực tế (đường nét đứt: mốc 70% và 85%)', fontweight='bold', y=1.04)
savefig(fig, '09_split_ratio_compare')

print(f'\nCác class có ≤ {RARE_MAX_GROUPS} nhóm ảnh đang rơi vào phần nào (số box):')
rare_place = rare[['class_name', 'n_groups', 'n_boxes']].set_index('class_name')
for mode in results:
    cb = reports[mode][1]
    rare_place = rare_place.join(cb.loc[rare_place.index].add_prefix(f'{mode}:'))
display(rare_place)
savetab(rare_place, 'rare_classes_placement', index=True)

# %% [markdown]
# ## 10. Xuất 2 file zip (định dạng YOLO, chưa tăng cường)
#
# Mỗi zip gồm `train/`, `valid/`, `test/` (mỗi phần có `images/` và `labels/`), `data.yaml`, `README_split.md` và thư mục `meta/` chứa bảng gán ảnh → nhóm → phần.
# File nhãn được ghi lại: dòng polygon đã đổi thành bbox; các dòng khác giữ nguyên.

# %%
def build_zip(mode):
    col = f'split_by_{mode}'
    root = BUILD_DIR / mode
    if root.exists():
        shutil.rmtree(root)
    for s in SPLITS:
        (root / s / 'images').mkdir(parents=True)
        (root / s / 'labels').mkdir(parents=True)
    for r in images.itertuples():
        s = getattr(r, col)
        shutil.copy2(MERGED_DIR / 'images' / r.image, root / s / 'images' / r.image)
        (root / s / 'labels' / f'{r.stem}.txt').write_text('\n'.join(label_lines[r.image]) + ('\n' if label_lines[r.image] else ''),
                                                           encoding='utf-8')
    cfg = {'train': 'train/images', 'val': 'valid/images', 'test': 'test/images', 'nc': NC, 'names': CLASS_NAMES}
    with open(root / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    (root / 'meta').mkdir()
    images[['image', 'group_id', 'group_size', 'dhash', col]].rename(columns={col: 'split'}).to_csv(
        root / 'meta' / 'image_group_split.csv', index=False, encoding='utf-8-sig')
    reports[mode][0].to_csv(root / 'meta' / 'split_overall.csv', encoding='utf-8-sig')
    reports[mode][1].add_prefix('box_').join(reports[mode][2].add_prefix('img_')).to_csv(
        root / 'meta' / 'split_per_class.csv', encoding='utf-8-sig')
    removed.to_csv(root / 'meta' / 'removed_images.csv', index=False, encoding='utf-8-sig')
    overall = reports[mode][0]
    readme = f"""# Bộ dữ liệu VN Traffic Sign – đã chia, chưa tăng cường

- Nguồn: cleaned_data.zip (Roboflow dataset-dap391m v4, YOLOv8), {NC} class theo QCVN 41:2024/BGTVT
- Gom nhóm ảnh gần trùng: dHash 64 bit (Zauner 2010; Vallez và cộng sự 2022), nối hai ảnh khi Hamming ≤ {DHASH_MAX_DIST}, nhóm = thành phần liên thông (union-find)
- Chia theo nhóm: iterative stratification (Sechidis, Tsoumakas, Vlahavas – ECML PKDD 2011) áp dụng ở cấp nhóm, phân tầng theo **số {'box' if mode == 'box' else 'ảnh chứa'}** mỗi class, tỉ lệ mục tiêu 70:15:15, seed {SEED}
- Kiểm tra rò rỉ: 0 cặp Hamming ≤ {DHASH_MAX_DIST} nằm ở hai phần khác nhau
- Nhãn: {n_poly_total} dòng polygon đã đổi thành bbox; {len(removed)} ảnh có class id ngoài 0–{NC - 1} đã bị loại
- Số nhóm: {gsize.size}; nhóm lớn nhất {int(gsize.max())} ảnh

| phần | số ảnh | % ảnh | số box | % box | số nhóm |
|---|---|---|---|---|---|
""" + '\n'.join(f"| {s} | {overall.loc[s, 'số ảnh']} | {overall.loc[s, '% ảnh']} | {overall.loc[s, 'số box']} | "
                f"{overall.loc[s, '% box']} | {overall.loc[s, 'số nhóm']} |" for s in SPLITS) + '\n'
    (root / 'README_split.md').write_text(readme, encoding='utf-8')

    local_zip = BUILD_DIR / f'split_data_by_{mode}.zip'
    with zipfile.ZipFile(local_zip, 'w') as zf:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                ctype = zipfile.ZIP_STORED if p.suffix.lower() in IMG_EXTS else zipfile.ZIP_DEFLATED
                zf.write(p, p.relative_to(root).as_posix(), compress_type=ctype)
    return local_zip


def verify_zip(zpath, mode):
    with zipfile.ZipFile(zpath) as zf:
        assert zf.testzip() is None
        nm = [n for n in zf.namelist() if not n.endswith('/')]
        seen = {}
        for n in nm:
            p = Path(n)
            if p.parent.name == 'images' and p.suffix.lower() in IMG_EXTS:
                assert p.name not in seen, f'{p.name} nằm ở 2 phần'
                seen[p.name] = p.parent.parent.name
                assert f'{p.parent.parent.name}/labels/{p.stem}.txt' in nm, f'Thiếu nhãn cho {n}'
    exp = images.set_index('image')[f'split_by_{mode}'].to_dict()
    assert seen == exp, 'Nội dung zip không khớp bảng chia'
    return len(seen)


for mode in results:
    z = build_zip(mode)
    n_ok = verify_zip(z, mode)
    dst = OUT_DIR / z.name
    shutil.copy2(z, dst)
    print(f'✅ {dst}  ({n_ok} ảnh, {dst.stat().st_size / 1e6:.1f} MB)')

images.drop(columns=['stem']).to_csv(TAB_DIR / 'image_group_split_both.csv', index=False, encoding='utf-8-sig')
summary = {
    'dhash_max_dist': DHASH_MAX_DIST, 'seed': SEED, 'ratios': RATIOS,
    'n_images': int(N), 'n_groups': int(gsize.size), 'largest_group': int(gsize.max()),
    'n_polygon_lines_converted': int(n_poly_total), 'n_images_removed_out_of_range': int(len(removed)),
    'n_classes_le2_groups': int(len(rare)),
    'compare': compare.to_dict(orient='records'),
}
with open(OUT_DIR / 'split_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print('\nCác file trong', OUT_DIR.name + ':')
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        print(f'  {p.name:32s} {p.stat().st_size / 1e6:9.2f} MB')
