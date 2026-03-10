# scripts/07_text_recognition.py
from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path

class BillRecognizer:
    def __init__(self, rec_model_dir=None, use_gpu=False):
        """
        初始化票据识别器
        Args:
            rec_model_dir: 自定义识别模型路径
            use_gpu: 是否使用GPU
        """
        # 票据场景优化参数[citation:1]
        self.ocr = PaddleOCR(
            use_gpu=use_gpu,
            rec_model_dir=rec_model_dir,
            rec_algorithm='SVTR_LCNet',  # 推荐垂直文本场景[citation:10]
            rec_batch_num=32,  # 批量识别，提升GPU利用率[citation:1]
            rec_image_shape='3, 48, 320',  # 适应票据文本高度[citation:10]
            use_space_char=True,  # 识别空格
            drop_score=0.5,  # 过滤低置信度结果
            show_log=False
        )
    
    def recognize_text(self, image_path, boxes=None):
        """
        识别文本内容
        Args:
            image_path: 图像路径
            boxes: 可选，指定检测框，否则先检测后识别
        Returns:
            results: 列表，每个元素为 (文本, 置信度, 坐标)
        """
        if boxes is None:
            # 执行完整OCR流程
            result = self.ocr.ocr(str(image_path), cls=True)
        else:
            # 使用指定的检测框进行识别
            img = cv2.imread(str(image_path))
            results = []
            for box in boxes:
                # 根据坐标裁剪区域
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                crop_img = img[y_min:y_max, x_min:x_max]
                
                # 临时保存裁剪图像
                temp_path = "temp_crop.jpg"
                cv2.imwrite(temp_path, crop_img)
                
                # 识别
                rec_result = self.ocr.ocr(temp_path, det=False, rec=True, cls=False)
                if rec_result and rec_result[0]:
                    text, score = rec_result[0][0]
                    results.append((text, score, box))
            
        return results

# 使用示例
if __name__ == "__main__":
    recognizer = BillRecognizer(use_gpu=False)
    results = recognizer.recognize_text("../data/enhanced/train/sample.jpg")
    
    for text, score, box in results:
        print(f"文本: {text}, 置信度: {score:.4f}")
        print(f"位置: {box}\n")
