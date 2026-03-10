# scripts/06_text_detection.py
import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
from pathlib import Path

class BillDetector:
    def __init__(self, det_model_dir=None, use_gpu=False):
        """
        初始化票据检测器
        Args:
            det_model_dir: 自定义检测模型路径（None表示使用预训练模型）
            use_gpu: 是否使用GPU
        """
        # 针对票据场景优化检测参数
        self.ocr = PaddleOCR(
            use_gpu=use_gpu,
            det_model_dir=det_model_dir,  # 可使用你在SCID上微调的模型
            det_db_thresh=0.22,  # 降低阈值提升小字号文字召回率[citation:10]
            det_db_box_thresh=0.55,  # 框过滤阈值
            det_db_unclip_ratio=2.0,  # 文本框扩展比例，适应倾斜文本[citation:10]
            det_db_score_mode='slow',  # 弯曲文本推荐使用'slow'[citation:4]
            show_log=False
        )
    
    def detect_text_boxes(self, image_path):
        """
        检测图像中的文本区域
        Returns:
            boxes: 文本框坐标列表，格式[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
            scores: 每个文本框的置信度
        """
        # 只执行检测，不执行识别
        result = self.ocr.ocr(str(image_path), det=True, rec=False, cls=False)
        
        if not result or not result[0]:
            return [], []
        
        boxes = []
        scores = []
        for line in result[0]:
            boxes.append(line[0])  # 坐标点
            scores.append(line[1][1])  # 置信度
        
        return boxes, scores
    
    def visualize_detection(self, image_path, boxes, output_path):
        """可视化检测结果"""
        img = cv2.imread(str(image_path))
        for box in boxes:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(str(output_path), img)

# 使用示例
if __name__ == "__main__":
    detector = BillDetector(use_gpu=False)
    boxes, scores = detector.detect_text_boxes("../data/enhanced/train/sample.jpg")
    print(f"检测到 {len(boxes)} 个文本区域")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"区域{i+1}: 坐标{box}, 置信度{score:.4f}")
