# scripts/09_bill_ocr_pipeline.py
import json
import time
from pathlib import Path
from scripts06_text_detection import BillDetector
from scripts07_text_recognition import BillRecognizer
from scripts08_kie_field_extraction import BillFieldExtractor

class BillOCRPipeline:
    """
    票据OCR完整流程
    """
    def __init__(self, 
                 det_model_dir=None,
                 rec_model_dir=None,
                 use_ser_model=False,
                 use_gpu=False):
        
        self.detector = BillDetector(det_model_dir, use_gpu)
        self.recognizer = BillRecognizer(rec_model_dir, use_gpu)
        self.extractor = BillFieldExtractor(use_ser_model)
        
    def process_single_image(self, image_path, visualize=False):
        """
        处理单张票据图像
        Returns:
            result: 结构化识别结果
        """
        start_time = time.time()
        
        # 步骤1：文本检测
        boxes, det_scores = self.detector.detect_text_boxes(image_path)
        det_time = time.time()
        
        # 步骤2：文本识别
        ocr_results = []
        for box in boxes:
            # 对每个检测框进行识别
            # 这里简化处理，实际应该用recognizer的批量识别
            img = cv2.imread(str(image_path))
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            crop = img[y_min:y_max, x_min:x_max]
            
            temp_path = "temp_crop.jpg"
            cv2.imwrite(temp_path, crop)
            rec_result = self.recognizer.ocr.ocr(temp_path, det=False, rec=True)
            
            if rec_result and rec_result[0]:
                text, conf = rec_result[0][0]
                ocr_results.append((text, conf, box))
        
        rec_time = time.time()
        
        # 步骤3：字段抽取
        fields, overall_conf = self.extractor.extract_fields(ocr_results, image_path)
        kie_time = time.time()
        
        # 组装结果
        result = {
            'file_name': str(image_path),
            'ocr_results': [
                {'text': t, 'confidence': c, 'box': b}
                for t, c, b in ocr_results
            ],
            'key_fields': fields,
            'overall_confidence': overall_conf,
            'performance': {
                'detection_time': det_time - start_time,
                'recognition_time': rec_time - det_time,
                'kie_time': kie_time - rec_time,
                'total_time': kie_time - start_time
            }
        }
        
        return result
    
    def batch_process(self, image_dir, output_json):
        """
        批量处理票据图像
        """
        image_paths = list(Path(image_dir).glob("*.jpg")) + \
                      list(Path(image_dir).glob("*.png"))
        
        all_results = []
        for img_path in image_paths[:10]:  # 先处理10张测试
            print(f"处理: {img_path.name}")
            result = self.process_single_image(img_path)
            all_results.append(result)
        
        # 保存结果
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 计算平均性能
        avg_time = np.mean([r['performance']['total_time'] for r in all_results])
        print(f"\n批量处理完成，平均单张耗时: {avg_time:.2f}秒")
        
        return all_results

# 主程序
if __name__ == "__main__":
    import cv2
    import numpy as np
    
    # 初始化pipeline
    pipeline = BillOCRPipeline(
        det_model_dir=None,  # 使用预训练模型
        rec_model_dir=None,  # 使用预训练模型
        use_ser_model=False,  # 先使用规则抽取
        use_gpu=False
    )
    
    # 处理单张图像
    result = pipeline.process_single_image("../data/enhanced/train/sample.jpg")
    
    # 输出结果
    print(json.dumps(result['key_fields'], ensure_ascii=False, indent=2))
    print(f"\n处理时间: {result['performance']['total_time']:.2f}秒")
