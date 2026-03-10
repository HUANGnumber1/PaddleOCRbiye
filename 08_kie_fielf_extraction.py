# scripts/08_kie_field_extraction.py
import json
import re
from pathlib import Path
import numpy as np

class BillFieldExtractor:
    """
    票据关键字段抽取器
    结合规则匹配和机器学习进行字段抽取
    """
    
    # 预定义字段类别（根据实际票据类型调整）
    FIELD_CATEGORIES = {
        'invoice_number': ['发票号码', '发票代码', 'No.', '号码'],
        'date': ['开票日期', '日期', 'Date', '开票日'],
        'amount': ['金额', '小写金额', '合计金额', '总计', 'Amount', 'Total'],
        'tax_amount': ['税额', '税率', 'Tax'],
        'seller': ['销售方', '卖方', '收款单位', 'Seller'],
        'buyer': ['购买方', '买方', '付款单位', 'Buyer']
    }
    
    def __init__(self, use_ser_model=False, ser_model_dir=None):
        """
        初始化字段抽取器
        Args:
            use_ser_model: 是否使用SER深度学习模型
            ser_model_dir: SER模型路径
        """
        self.use_ser_model = use_ser_model
        
        if use_ser_model:
            # 加载PaddleOCR的SER模型（VI-LayoutXLM）[citation:8]
            from paddleocr import PPStructure
            self.ser_engine = PPStructure(
                table=False,
                ocr=False,
                mode='kie',
                image_orientation=True,
                layout_model_dir=ser_model_dir
            )
    
    def extract_by_rules(self, ocr_results):
        """
        基于规则的字段抽取
        Args:
            ocr_results: OCR识别结果列表，每个元素为(text, confidence, box)
        Returns:
            fields: 抽取的关键字段字典
        """
        fields = {
            'invoice_number': {'value': None, 'confidence': 0, 'source': None},
            'date': {'value': None, 'confidence': 0, 'source': None},
            'amount': {'value': None, 'confidence': 0, 'source': None},
            'tax_amount': {'value': None, 'confidence': 0, 'source': None},
            'seller': {'value': None, 'confidence': 0, 'source': None},
            'buyer': {'value': None, 'confidence': 0, 'source': None}
        }
        
        # 1. 基于关键词匹配
        for text, conf, box in ocr_results:
            text_clean = text.strip()
            
            for field_key, keywords in self.FIELD_CATEGORIES.items():
                # 检查是否包含关键词
                for keyword in keywords:
                    if keyword in text_clean:
                        # 提取字段值（通常是关键词后面的部分）
                        value = text_clean.replace(keyword, '').strip()
                        if value and conf > fields[field_key]['confidence']:
                            fields[field_key]['value'] = value
                            fields[field_key]['confidence'] = conf
                            fields[field_key]['source'] = 'keyword_match'
                        break
        
        # 2. 正则表达式精准匹配
        # 发票号：通常为数字+字母组合
        invoice_pattern = r'[A-Z0-9]{8,20}'
        # 日期：YYYY-MM-DD 或 YYYY年MM月DD日
        date_pattern = r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?'
        # 金额：数字+小数点+两位小数
        amount_pattern = r'[0-9,]+\.\d{2}'
        
        for text, conf, box in ocr_results:
            # 发票号匹配
            if not fields['invoice_number']['value']:
                match = re.search(invoice_pattern, text)
                if match:
                    fields['invoice_number']['value'] = match.group()
                    fields['invoice_number']['confidence'] = conf
                    fields['invoice_number']['source'] = 'regex_match'
            
            # 日期匹配
            if not fields['date']['value']:
                match = re.search(date_pattern, text)
                if match:
                    fields['date']['value'] = match.group()
                    fields['date']['confidence'] = conf
                    fields['date']['source'] = 'regex_match'
            
            # 金额匹配
            if not fields['amount']['value']:
                match = re.search(amount_pattern, text)
                if match:
                    fields['amount']['value'] = match.group()
                    fields['amount']['confidence'] = conf
                    fields['amount']['source'] = 'regex_match'
        
        return fields
    
    def extract_by_ser_model(self, image_path):
        """
        基于SER深度学习模型的字段抽取[citation:8]
        Args:
            image_path: 图像路径
        Returns:
            fields: SER模型输出的字段分类结果
        """
        if not self.use_ser_model:
            return None
        
        # 使用PPStructure的KIE能力
        result = self.ser_engine(image_path)
        
        fields = {}
        for item in result:
            if item['type'] == 'kie':
                # 解析SER结果
                for region in item['res']:
                    field_name = region['label']  # 如 'invoice_number'
                    field_value = region['text']
                    field_conf = region['confidence']
                    
                    fields[field_name] = {
                        'value': field_value,
                        'confidence': field_conf,
                        'bbox': region.get('bbox', [])
                    }
        
        return fields
    
    def extract_fields(self, ocr_results, image_path=None):
        """
        综合抽取字段
        Returns:
            fields: 结构化字段结果
            overall_confidence: 整体置信度
        """
        # 优先使用规则抽取
        rule_fields = self.extract_by_rules(ocr_results)
        
        # 如果有SER模型，可以融合结果
        if self.use_ser_model and image_path:
            ser_fields = self.extract_by_ser_model(image_path)
            # 这里可以实现规则和模型的融合策略
            # 例如：取置信度高的结果，或规则优先等
        
        # 计算整体置信度（所有抽取字段的平均置信度）
        confidences = [f['confidence'] for f in rule_fields.values() if f['value']]
        overall_conf = np.mean(confidences) if confidences else 0
        
        return rule_fields, overall_conf

# 使用示例
if __name__ == "__main__":
    # 模拟OCR结果
    ocr_results = [
        ("发票号码：INV20260310", 0.99, [[120,450],[380,450],[380,480],[120,480]]),
        ("开票日期：2026-03-10", 0.98, [[120,500],[380,500],[380,530],[120,530]]),
        ("金额合计：￥1,200.00", 0.97, [[120,550],[380,550],[380,580],[120,580]]),
        ("销售方：XX科技有限公司", 0.96, [[120,600],[380,600],[380,630],[120,630]]),
    ]
    
    extractor = BillFieldExtractor(use_ser_model=False)
    fields, confidence = extractor.extract_fields(ocr_results)
    
    print("=== 抽取的关键字段 ===")
    for field_name, field_info in fields.items():
        if field_info['value']:
            print(f"{field_name}: {field_info['value']} (置信度: {field_info['confidence']:.2f})")
    print(f"\n整体置信度: {confidence:.2f}")
