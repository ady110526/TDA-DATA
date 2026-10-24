# Hướng dẫn chạy huấn luyện và đánh giá (đúng như quy trình trong bài báo)

## Phân công
| File | Máy trong bài | Mô hình |
|---|---|---|
| `train_may1_rtx5050.py` | RTX 5050 Laptop 8 GB | Faster R-CNN ResNet50-FPN, YOLO26s |
| `train_may2_rtx4050.py` | RTX 4050 Laptop 6 GB | YOLOv8s, YOLO11s |
| `train_may3_rtx3050.py` | RTX 3050 Laptop 6 GB | YOLOv8n, YOLO11n, YOLO26n |

Mỗi file kiểm tra tên GPU; chạy trên GPU khác thì thêm `--force-gpu`. Ba file dùng chung `common.py` và `config.yaml` – hai file này phải giống hệt nhau trên mọi máy:
- `common.py`: mã kiểm tra `c782e2b9b3960e70`
- `config.yaml`: `config_sha256 = 5c66a7c61b5b27dd` (không sửa cả phần chú thích, vì mã băm tính trên toàn bộ tệp)

Kiểm tra mã của `common.py`:
```
python -c "import hashlib;print(hashlib.sha256(open('common.py','rb').read().replace(b'\r\n',b'\n')).hexdigest()[:16])"
```

## Bước 1 – Môi trường
```
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Bước 2 – Dữ liệu
Giải nén `split_data_by_box.zip` (đầu ra của notebook `03_group_split`) vào thư mục không dấu, ví dụ `D:\VNTrafficSign\split_data_by_box`. Không thêm, xóa, sửa file trong thư mục này.

## Bước 3 – Kiểm tra
```
python train_mayX_....py --data <thư mục dữ liệu> --check
```
Phải in ra `config_sha256 = 5c66a7c61b5b27dd`, `dataset_fingerprint = e763e77b581d07df`, và train/valid/test = 10.100 / 2.203 / 2.206 ảnh.

## Bước 4 – Chạy thử VRAM
```
python train_mayX_....py --data <thư mục dữ liệu> --smoke
```
Hết bộ nhớ GPU thì **không** tự giảm batch trên một máy (batch là một phần cấu hình chung: YOLO 16, Faster R-CNN 4).

## Bước 5 – Hiệu chuẩn (mọi máy)
```
python train_mayX_....py --data <thư mục dữ liệu> --calib
```
Huấn luyện YOLOv8n seed 0 với cùng cấu hình để đo riêng mức lệch do khác máy.

## Bước 6 – Huấn luyện
```
python train_mayX_....py --data <thư mục dữ liệu>
```
- Bị ngắt thì chạy lại đúng lệnh: code chạy tiếp từ checkpoint gần nhất và bỏ qua các lần đã xong.
- Máy thiếu RAM (lỗi `DataLoader worker exited unexpectedly` hoặc `paging file is too small`): thêm `--workers 2` hoặc `--workers 0`; giá trị được ghi vào `env.json`. Trên Windows có thể tăng paging file (16–32 GB).
- Kết quả mỗi lần chạy: `results/<mô hình>/seed<k>/` (`metrics.json`, `per_class_AP_test.csv`, `test_predictions_coco.json`, `env.json`, `config_used.yaml`, nhật ký). Bảng tóm tắt: `results/summary.csv`.

## Bước 7 – Dự đoán tập valid (để chọn ngưỡng F1)
```
python predict_valid.py --data <thư mục dữ liệu>
```
Ghi `valid_predictions_coco.json` cạnh `test_predictions_coco.json` của mỗi lần chạy, và `valid_gt_coco.json`.

## Bước 8 – Tổng hợp mAP
Gom thư mục `results` của các máy về một chỗ (`results_may1`, `results_may2`, `results_may3`):
```
python common.py aggregate results_may1 results_may2 results_may3
```

## Bước 9 – Đo tốc độ (một máy duy nhất, sau khi huấn luyện xong)
Đặt `best.pt`/`best.pth` của seed 0 mọi mô hình vào các thư mục kết quả trên máy đo (giữ cấu trúc `<mô hình>/seed0/...`), cắm sạc, chế độ hiệu năng cao, đóng ứng dụng khác:
```
python benchmark_fps.py --data <thư mục dữ liệu> --results results results_may2 results_may3
```
Kết quả `fps/fps_results.csv`; bài báo dùng **trung vị** ms/ảnh (`ms_median`). Cột `fp16_active` xác nhận FP16 thật sự được bật.

## Bước 10 – Nhãn thật cho phân tích F1 / ma trận nhầm lẫn
```
python export_gt.py --data <thư mục dữ liệu> --split test
python export_gt.py --data <thư mục dữ liệu> --split valid
```
Sau đó dùng các script trong `../evaluation/` (xem README).

## Hai họ mô hình dùng công thức chuẩn riêng
- YOLO: công thức mặc định Ultralytics (ảnh 640, 100 epoch, `optimizer=auto` → MuSGD, mosaic, HSV…), chỉ tắt lật ngang (`fliplr = 0`) vì có nhiều cặp biển trái/phải.
- Faster R-CNN: công thức tham chiếu torchvision (cạnh ngắn 800, SGD lr 0,005 cho batch 4, 26 epoch, giảm lr ở epoch 16 và 22, không tăng cường, không lật ngang).
