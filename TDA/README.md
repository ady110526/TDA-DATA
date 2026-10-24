# TDA – Bộ dữ liệu phát hiện biển báo giao thông Việt Nam theo QCVN 41:2024

Mã nguồn của bài báo *“TDA: Bộ dữ liệu phát hiện biển báo giao thông Việt Nam theo QCVN 41:2024 với phép chia tập chống rò rỉ dữ liệu và benchmark các mô hình phát hiện đối tượng”*.

- Bộ dữ liệu: 14.509 ảnh, 27.150 hộp giới hạn, 115 lớp (5 nhóm biển theo QCVN 41:2024).
- Chia tập theo **nhóm ảnh gần trùng** (dHash, Hamming ≤ 2) bằng **phân tầng lặp theo số hộp** → 70 : 15 : 15, không có cặp ảnh gần trùng nào nằm ở hai tập khác nhau.
- Benchmark 7 mô hình: YOLOv8 / YOLO11 / YOLO26 (cỡ n, s; mỗi mô hình 3 seed) và Faster R-CNN ResNet50-FPN (1 seed), đánh giá chung bằng pycocotools.

Link bộ dữ liệu: `<điền link>` · Bài báo: `<điền link khi có>`

## Cấu trúc thư mục

| Thư mục | Nội dung | Chạy ở đâu |
|---|---|---|
| `data_pipeline/` | 3 notebook xử lý dữ liệu (kèm bản `.py` để dễ đọc/so sánh) | Google Colab |
| `training/` | Huấn luyện, đánh giá mAP trên test, dự đoán tập valid, đo tốc độ | Máy có GPU NVIDIA (Windows/Linux) |
| `evaluation/` | F1 (ngưỡng chọn trên valid), ma trận nhầm lẫn, phân tích theo lớp, kiểm tra heatmap | Máy bất kỳ, không cần GPU |

## 1. Xử lý dữ liệu (`data_pipeline/`, Google Colab)

| Notebook | Việc làm | Đầu ra chính |
|---|---|---|
| `01_clean_data.ipynb` | Loại ảnh không có tệp nhãn hoặc tệp nhãn rỗng (14.756 → 14.509 ảnh) | `cleaned_data.zip` |
| `02_eda.ipynb` | Phân tích dữ liệu: phân bố lớp/nhóm, kích thước, vị trí, đồng xuất hiện, kiểm tra toàn vẹn; nhãn đa giác được đổi thành hộp bao ngoài | `output_eda/` (bảng + hình) |
| `03_group_split.ipynb` | dHash → nối cặp Hamming ≤ 2 → union-find → phân tầng lặp ở cấp nhóm (theo số hộp và theo số ảnh) → kiểm tra rò rỉ | `split_data_by_box.zip` (bản dùng trong bài), `split_data_by_image.zip`, `output_split/tables/` |

Đường dẫn Google Drive khai báo ở ô đầu mỗi notebook. Notebook giải nén vào `/content`, không giải nén lên Drive.

## 2. Huấn luyện và đánh giá (`training/`)

### Môi trường (giống nhau trên mọi máy)
```
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Phiên bản khóa: PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0, Ultralytics 8.4.178, pycocotools 2.0.11. Code dừng nếu phiên bản lệch.

### Kiểm tra trước khi chạy
```
python train_may1_rtx5050.py --data <thư mục split_data_by_box> --check
```
Phải in ra `config_sha256 = 5c66a7c61b5b27dd` và `dataset_fingerprint = e763e77b581d07df`.

### Các lệnh
| Lệnh | Việc làm |
|---|---|
| `python train_mayX_....py --data <dir> --smoke` | Chạy thử 1 epoch trên 5% dữ liệu, in VRAM tối đa |
| `python train_mayX_....py --data <dir>` | Huấn luyện tất cả mô hình × seed của máy đó, rồi chấm mAP trên test (pycocotools). Bị ngắt thì chạy lại đúng lệnh, code tự chạy tiếp |
| `... --only <mô hình> --seed <k>` | Chỉ chạy một mô hình / một seed |
| `... --calib` | Hiệu chuẩn giữa các máy: YOLOv8n seed 0 |
| `... --workers N` | Giảm số tiến trình đọc dữ liệu trên máy thiếu RAM (không đổi `config_sha256`, được ghi vào `env.json`) |
| `... --force-gpu` | Cho phép chạy file của máy khác (ví dụ chạy Faster R-CNN bằng `train_may1_rtx5050.py` trên GPU khác) |
| `python common.py aggregate results_may1 results_may2 results_may3` | Gom kết quả 3 máy → trung bình ± độ lệch chuẩn, bảng hiệu chuẩn |
| `python predict_valid.py --data <dir>` | Dự đoán tập valid cho mọi lần huấn luyện (để chọn ngưỡng F1 trên valid) |
| `python export_gt.py --data <dir> --split test` (và `--split valid`) | Xuất nhãn thật dạng COCO cho `evaluation/` |
| `python benchmark_fps.py --data <dir> --results results results_may2 results_may3` | Đo tốc độ 7 mô hình trên một máy: batch 1, 300 ảnh test, 50 ảnh làm nóng, FP32 và FP16 |
| `python xuat_anh_nham_lan.py --data <dir> --pred results/yolo26s/seed1/test_predictions_coco.json --conf 0.37` | Cắt ảnh từng trường hợp nhầm lớp và từng trường hợp trong các ô ma trận nhầm lẫn có ≥ 10 ca (bỏ sót, báo nhầm), kèm chẩn đoán tự động, để xem lại bằng mắt |

Phân công trong bài: máy 1 (RTX 5050 8 GB) – YOLO26s, Faster R-CNN; máy 2 (RTX 4050 6 GB) – YOLOv8s, YOLO11s; máy 3 (RTX 3050 6 GB) – YOLOv8n, YOLO11n, YOLO26n. Chi tiết từng bước: `training/HUONG_DAN.md`.

Kết quả mỗi lần chạy nằm ở `results/<mô hình>/seed<k>/`: `metrics.json`, `per_class_AP_test.csv`, `test_predictions_coco.json`, `valid_predictions_coco.json`, `env.json`, `config_used.yaml`, nhật ký huấn luyện.

## 3. Phân tích thêm (`evaluation/`, không cần GPU)

Gom thư mục `results` của 3 máy về một chỗ (`results_may1`, `results_may2`, `results_may3`), cùng `test_gt_coco.json`, `valid_gt_coco.json` và các bảng của notebook 02/03, rồi:
```
pip install -r requirements.txt
python f1_valid_threshold.py --valid-gt valid_gt_coco.json --test-gt test_gt_coco.json --results results_may1 results_may2 results_may3 --out f1_out
python confusion_report.py --cm-dir f1_out --class-map class_group_mapping.csv --models yolo26s frcnn_r50fpn --out cm_out
python per_class_ap.py --results results_may1 results_may2 results_may3 --class-dist class_distribution.csv --test-gt test_gt_coco.json
python heatmap_group_check.py --boxes boxes.csv --groups image_group_split_both.csv
python ma_tran_nham_lan_115.py --cm f1_out/cm_yolo26s_seed1.npy --class-map class_group_mapping.csv --mode phan_tram --out 16_confusion_115_yolo26s_seed1.png
python phan_tich_nguyen_nhan.py --gt test_gt_coco.json --pred results_may1/yolo26s/seed1/test_predictions_coco.json --conf 0.37 --class-map class_group_mapping.csv
```
- F1: với mỗi lần huấn luyện, dò ngưỡng tin cậy 0,05–0,95 (bước 0,01) trên **valid**, chọn ngưỡng cho F1 vi mô cao nhất, áp nguyên ngưỡng đó lên **test**; ghép dự đoán với hộp thật cùng lớp, IoU ≥ 0,5. Kèm F1 ở ngưỡng cố định 0,25 để tham khảo.
- Ma trận nhầm lẫn: cùng ngưỡng chọn trên valid, ghép không phân biệt lớp, cột “nền” = bỏ sót, hàng “nền” = dự đoán không khớp hộp thật nào.
- Hình ma trận nhầm lẫn 115 lớp trong bài là của **YOLO26s seed 1** (lần có mAP cao nhất, ngưỡng 0,37 chọn trên valid), vẽ bằng `ma_tran_nham_lan_115.py`; các con số giải thích nguyên nhân (bỏ sót theo kích thước hộp, chiều nhầm lẫn…) in ra bằng `phan_tich_nguyen_nhan.py`. Với Faster R-CNN dùng `--conf 0.67`.
- `confusion_report.py` cộng dồn 3 seed của mỗi mô hình (ma trận 115 lớp dạng CSV, bảng cặp nhầm, ma trận theo 5 nhóm); dùng cho các số liệu cộng dồn ở mục “Kết quả theo lớp”.
- `class_group_mapping.csv`, `class_distribution.csv`, `boxes.csv`: bảng do notebook `02_eda` xuất ra (`output_eda/tables/`); `image_group_split_both.csv`: bảng do notebook `03_group_split` xuất ra.

## Kết quả chính (tập test)

| Mô hình | mAP@[.5:.95] | mAP@.5 | F1 (ngưỡng chọn trên valid) | ms/ảnh FP32 (RTX 5050) |
|---|---|---|---|---|
| YOLOv8n | 62,66 ± 0,30 | 81,10 ± 0,66 | 84,25 ± 0,46 | 6,71 |
| YOLO11n | 61,98 ± 0,76 | 79,44 ± 0,78 | 83,40 ± 0,11 | 8,12 |
| YOLO26n | 63,94 ± 0,50 | 81,59 ± 0,67 | 85,68 ± 0,44 | 9,17 |
| YOLOv8s | 67,72 ± 0,47 | 85,91 ± 0,56 | 87,38 ± 0,22 | 7,01 |
| YOLO11s | 67,14 ± 0,41 | 84,82 ± 0,58 | 87,30 ± 0,16 | 8,24 |
| YOLO26s | **69,45 ± 1,17** | **87,00 ± 1,32** | **88,69 ± 0,03** | 9,25 |
| Faster R-CNN R50-FPN (1 seed) | 62,12 | 80,21 | 83,12 | 78,19 |

Hiệu chuẩn (YOLOv8n seed 0, cùng cấu hình): RTX 5050 62,26 · RTX 4050 63,40 · RTX 3050 62,94 mAP@[.5:.95].

## Ghi chú tái lập
- Trên cùng một máy, YOLO cho kết quả giống hệt nhau giữa hai lần chạy (đã kiểm chứng trên RTX 3050). Faster R-CNN không tất định tuyệt đối do ROI Align của torchvision.
- Khác GPU thì kết quả lệch nhẹ; mức lệch được đo bằng `--calib`.
- Các sai khác so với cấu hình chuẩn trong bài (số tiến trình đọc dữ liệu trên máy thiếu RAM, các lần chạy tiếp từ checkpoint) được ghi đầy đủ trong mục “Phần cứng và giao thức tái lập” của bài báo và trong `env.json` / `args.yaml` của từng lần chạy.

## Giấy phép
Mã nguồn: `<chọn giấy phép, ví dụ MIT>`. Bộ dữ liệu: xem link bộ dữ liệu.

## Trích dẫn
```
<điền BibTeX khi bài báo được chấp nhận>
```
