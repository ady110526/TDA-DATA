"""
MÁY 2 – RTX 4050 Laptop (Windows)
Mô hình: YOLOv8s, YOLO11s

Lưu ý: nhóm ghi "7.9 GB"; RTX 4050 Laptop thường có 6 GB VRAM. Chạy `nvidia-smi` để biết chính xác.
Batch dùng chung (config.yaml → yolo.batch) đã chọn để vừa máy 6 GB; kiểm tra bằng --smoke.

Cách chạy (terminal trong VS Code, đã kích hoạt môi trường chung):
    python train_may2_rtx4050.py --data D:\\VNTrafficSign\\split_data_by_box --check
    python train_may2_rtx4050.py --data D:\\VNTrafficSign\\split_data_by_box --smoke
    python train_may2_rtx4050.py --data D:\\VNTrafficSign\\split_data_by_box --calib
    python train_may2_rtx4050.py --data D:\\VNTrafficSign\\split_data_by_box
Chỉ một mô hình / một seed:  --only yolo11s   --seed 0
"""
import common

MACHINE = {
    'id': 'may2_rtx4050',
    'label': 'Máy 2 – RTX 4050 Laptop',
    'gpu_hint': '4050',
    'models': ['yolov8s', 'yolo11s'],
}

if __name__ == '__main__':
    common.main(MACHINE)
