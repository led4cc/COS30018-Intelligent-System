import time
import torch
import cv2
from numpy import random
from models.experimental import attempt_load
from utils.datasets import transform_img
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized

def detect(model, image, device, imgsz=640, conf_thres=0.25, iou_thres=0.45, augment=False, classes=0, agnostic_nms=False):
    # Initialize
    half = False  # Disable half precision to avoid type mismatch

    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # Ensure model is in float32 precision
    model.float()

    img, im0 = transform_img(image)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.float()  # Ensure input is in float32 precision
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    t2 = time_synchronized()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

    print("Prediction results:", pred)  # Debug statement

    final_pred = []

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            final_pred.append(det)
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        print('Number of License Plate:', len(det))

    print(f'Done. ({time.time() - t0:.3f}s)')
    return final_pred, im0
