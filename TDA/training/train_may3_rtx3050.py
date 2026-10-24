"""
MÁY 3 – RTX 3050 6 GB (Windows)
Mô hình: YOLOv8n, YOLO11n, YOLO26n

Máy yếu nhất nhận các mô hình cỡ n (nhẹ nhất). Đây cũng là máy dùng để kiểm tra
batch dùng chung bằng --smoke trước khi cả nhóm train thật.

Cách chạy (terminal trong VS Code, đã kích hoạt môi trường chung):
    python train_may3_rtx3050.py --data D:\\VNTrafficSign\\split_data_by_box --check
    python train_may3_rtx3050.py --data D:\\VNTrafficSign\\split_data_by_box --smoke
    python train_may3_rtx3050.py --data D:\\VNTrafficSign\\split_data_by_box --calib
    python train_may3_rtx3050.py --data D:\\VNTrafficSign\\split_data_by_box
Chỉ một mô hình / một seed:  --only yolo26n   --seed 0
"""
import common

MACHINE = {
    'id': 'may3_rtx3050',
    'label': 'Máy 3 – RTX 3050 6 GB',
    'gpu_hint': '3050',
    'models': ['yolov8n', 'yolo11n', 'yolo26n'],
}

if __name__ == '__main__':
    common.main(MACHINE)
