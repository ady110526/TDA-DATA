"""
MÁY 1 – RTX 5050 8 GB (Windows)
Mô hình: Faster R-CNN ResNet50-FPN, YOLO26s

Faster R-CNN đặt ở máy này vì nặng nhất (ảnh 800 px) và cần nhiều VRAM nhất.

Cách chạy (terminal trong VS Code, đã kích hoạt môi trường chung):
    python train_may1_rtx5050.py --data D:\\VNTrafficSign\\split_data_by_box --check
    python train_may1_rtx5050.py --data D:\\VNTrafficSign\\split_data_by_box --smoke
    python train_may1_rtx5050.py --data D:\\VNTrafficSign\\split_data_by_box --calib
    python train_may1_rtx5050.py --data D:\\VNTrafficSign\\split_data_by_box
Chỉ một mô hình / một seed:  --only yolo26s   --seed 0
"""
import common

MACHINE = {
    'id': 'may1_rtx5050',
    'label': 'Máy 1 – RTX 5050 8 GB',
    'gpu_hint': '5050',          # tên GPU phải chứa chuỗi này
    'models': ['frcnn_r50fpn', 'yolo26s'],
}

if __name__ == '__main__':      # bắt buộc trên Windows (DataLoader nhiều worker)
    common.main(MACHINE)
