import json
import os
import re
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher
import argparse

class GroundTruthValidator:
    def __init__(self, annotations_path: str):
        """Khởi tạo validator với file annotations gốc"""
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # Tạo mapping từ image ID sang annotation
        self.annotation_map = {str(item['id']): item for item in self.annotations}

        # Validation rules
        self.validation_rules = {
            'date_format': r'\d{4}-\d{2}-\d{2}',
            'numeric_fields': ['total_payment', 'quantity', 'unit_price'],
            'required_fields': ['document_type', 'general_info', 'financials']
        }

    def extract_info_from_annotations(self, annotation: Dict) -> Dict[str, Any]:
        """Trích xuất thông tin quan trọng từ annotation gốc"""
        description = annotation.get('description', '')
        extractions = annotation.get('extractions', '{}')

        try:
            # Parse extractions JSON string
            if isinstance(extractions, str):
                extractions = json.loads(extractions.replace("'", '"'))
            elif not isinstance(extractions, dict):
                extractions = {}
        except:
            extractions = {}

        return {
            'description': description,
            'extractions': extractions,
            'seller': extractions.get('Tên cửa hàng', ''),
            'date': extractions.get('Ngày bán', ''),
            'invoice_code': extractions.get('Số hoá đơn', ''),
            'total_amount': extractions.get('Tổng tiền thanh toán', ''),
            'buyer': extractions.get('Tên khách hàng', '')
        }

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Tính độ tương đồng giữa hai chuỗi text"""
        if not text1 or not text2:
            return 0.0

        # Normalize text
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        return SequenceMatcher(None, text1, text2).ratio()

    def validate_date_format(self, date_str: str) -> bool:
        """Kiểm tra format ngày tháng"""
        if not date_str:
            return False
        return bool(re.match(self.validation_rules['date_format'], str(date_str)))

    def validate_numeric_fields(self, data: Dict) -> Dict[str, Any]:
        """Validate các field số"""
        issues = {}

        def check_numeric(value, field_path):
            if value is not None:
                try:
                    if isinstance(value, str):
                        # Remove commas and convert
                        clean_value = value.replace(',', '').replace('.', '')
                        float(clean_value)
                    else:
                        float(value)
                except (ValueError, TypeError):
                    issues[field_path] = f"Invalid numeric value: {value}"

        # Check financials
        if 'financials' in data:
            financials = data['financials']
            check_numeric(financials.get('total_payment'), 'financials.total_payment')

        # Check items
        if 'items' in data and isinstance(data['items'], list):
            for i, item in enumerate(data['items']):
                check_numeric(item.get('quantity'), f'items[{i}].quantity')
                check_numeric(item.get('unit_price'), f'items[{i}].unit_price')
                check_numeric(item.get('total_money'), f'items[{i}].total_money')

        return issues

    def validate_ground_truth(self, ground_truth: Dict, annotation_id: str) -> Dict[str, Any]:
        """Validate ground truth với annotation gốc"""
        if annotation_id not in self.annotation_map:
            return {
                'valid': False,
                'error': f'Annotation ID {annotation_id} not found',
                'issues': []
            }

        annotation = self.annotation_map[annotation_id]
        annotation_info = self.extract_info_from_annotations(annotation)

        issues = []
        scores = {}

        # Validate required fields
        for field in self.validation_rules['required_fields']:
            if field not in ground_truth:
                issues.append(f"Missing required field: {field}")

        # Validate seller/manufacturer
        gt_seller = ground_truth.get('general_info', {}).get('seller_or_manufacturer', '')
        ann_seller = annotation_info['seller']
        seller_similarity = self.calculate_text_similarity(gt_seller, ann_seller)
        scores['seller_match'] = seller_similarity

        if seller_similarity < 0.6:
            issues.append(f"Low seller similarity: {seller_similarity:.2f} - GT: '{gt_seller}' vs ANN: '{ann_seller}'")

        # Validate date
        gt_date = ground_truth.get('general_info', {}).get('date', '')
        ann_date = annotation_info['date']
        date_similarity = self.calculate_text_similarity(str(gt_date), str(ann_date))
        scores['date_match'] = date_similarity

        if not self.validate_date_format(str(gt_date)):
            issues.append(f"Invalid date format: {gt_date}")

        if date_similarity < 0.8:
            issues.append(f"Low date similarity: {date_similarity:.2f} - GT: '{gt_date}' vs ANN: '{ann_date}'")

        # Validate invoice code
        gt_code = ground_truth.get('general_info', {}).get('code', '')
        ann_code = annotation_info['invoice_code']
        code_similarity = self.calculate_text_similarity(gt_code, ann_code)
        scores['code_match'] = code_similarity

        if code_similarity < 0.7:
            issues.append(f"Low code similarity: {code_similarity:.2f} - GT: '{gt_code}' vs ANN: '{ann_code}'")

        # Validate total amount
        gt_total = ground_truth.get('financials', {}).get('total_payment')
        ann_total = annotation_info['total_amount']
        total_similarity = self.calculate_text_similarity(str(gt_total), str(ann_total))
        scores['total_match'] = total_similarity

        if total_similarity < 0.8:
            issues.append(f"Low total similarity: {total_similarity:.2f} - GT: '{gt_total}' vs ANN: '{ann_total}'")

        # Validate numeric fields
        numeric_issues = self.validate_numeric_fields(ground_truth)
        issues.extend([f"{field}: {msg}" for field, msg in numeric_issues.items()])

        # Calculate overall quality score
        quality_score = sum(scores.values()) / len(scores) if scores else 0

        return {
            'valid': len(issues) == 0,
            'quality_score': quality_score,
            'issues': issues,
            'scores': scores,
            'annotation_info': annotation_info
        }

    def validate_batch(self, ground_truth_dir: str, image_ids: List[str] = None) -> Dict[str, Any]:
        """Validate batch of ground truth files"""
        results = {}

        if image_ids is None:
            # Extract IDs from filenames
            image_ids = []
            for filename in os.listdir(ground_truth_dir):
                if filename.endswith('.json'):
                    base_name = filename.replace('.json', '')
                    # Try to extract ID from filename (assuming format like image_90.json)
                    match = re.search(r'image_(\d+)', base_name)
                    if match:
                        image_ids.append(match.group(1))

        for image_id in image_ids:
            gt_file = os.path.join(ground_truth_dir, f'image_{image_id}.json')
            if os.path.exists(gt_file):
                try:
                    with open(gt_file, 'r', encoding='utf-8') as f:
                        ground_truth = json.load(f)

                    result = self.validate_ground_truth(ground_truth, image_id)
                    results[image_id] = result

                    status = "✓" if result['valid'] else "✗"
                    print(f"{status} image_{image_id}.json - Quality: {result['quality_score']:.2f} - Issues: {len(result['issues'])}")

                except Exception as e:
                    results[image_id] = {'error': str(e)}
                    print(f"✗ image_{image_id}.json - Error: {e}")

        return results

def main():
    parser = argparse.ArgumentParser(description='Validate ground truth JSONs against original annotations')
    parser.add_argument('--annotations', default='mc_ocr_dataset/annotations/annotations.json',
                       help='Path to original annotations file')
    parser.add_argument('--ground_truth_dir', required=True,
                       help='Directory containing ground truth JSON files')
    parser.add_argument('--output', help='Output file for validation results')

    args = parser.parse_args()

    validator = GroundTruthValidator(args.annotations)
    results = validator.validate_batch(args.ground_truth_dir)

    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")

    # Summary
    total_files = len(results)
    valid_files = sum(1 for r in results.values() if isinstance(r, dict) and r.get('valid', False))
    avg_quality = sum(r.get('quality_score', 0) for r in results.values() if isinstance(r, dict)) / total_files

    print("
=== VALIDATION SUMMARY ===")
    print(f"Total files: {total_files}")
    print(f"Valid files: {valid_files} ({valid_files/total_files*100:.1f}%)")
    print(f"Average quality score: {avg_quality:.3f}")

if __name__ == '__main__':
    main()
