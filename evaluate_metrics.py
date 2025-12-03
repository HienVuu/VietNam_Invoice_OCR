import json
import re
from difflib import SequenceMatcher
from datetime import datetime

def normalize_number(value):
    """Normalize number by removing commas and converting to float."""
    if isinstance(value, str):
        value = re.sub(r'[^\d.]', '', value)
        try:
            return float(value)
        except ValueError:
            return None
    return value

def normalize_date(date_str):
    """Normalize date to YYYY-MM-DD format."""
    if not date_str:
        return None
    # Common formats: DD/MM/YYYY, MM/DD/YYYY, etc.
    patterns = [
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY or MM/DD/YYYY
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
    ]
    for pattern in patterns:
        match = re.match(pattern, date_str)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:  # YYYY-MM-DD
                return f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
            else:  # Assume DD/MM/YYYY or MM/DD/YYYY, but since Vietnamese, likely DD/MM/YYYY
                day, month, year = groups
                try:
                    # Validate date
                    datetime(int(year), int(month), int(day))
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    continue
    return None

def levenshtein_similarity(a, b):
    """Calculate Levenshtein similarity ratio."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def evaluate_hard_fields(pred, gt):
    """Evaluate Hard Fields: exact match for total_payment, date, tax_id."""
    scores = {}
    # total_payment
    pred_total = normalize_number(pred.get('financials', {}).get('total_payment', {}).get('value') if isinstance(pred.get('financials', {}).get('total_payment'), dict) else pred.get('financials', {}).get('total_payment'))
    gt_total = normalize_number(gt.get('financials', {}).get('total_payment', {}).get('value') if isinstance(gt.get('financials', {}).get('total_payment'), dict) else gt.get('financials', {}).get('total_payment'))
    scores['total_payment'] = 1.0 if pred_total == gt_total and pred_total is not None else 0.0

    # date
    pred_date = normalize_date(pred.get('general_info', {}).get('date', {}).get('value') if isinstance(pred.get('general_info', {}).get('date'), dict) else pred.get('general_info', {}).get('date'))
    gt_date = normalize_date(gt.get('general_info', {}).get('date', {}).get('value') if isinstance(gt.get('general_info', {}).get('date'), dict) else gt.get('general_info', {}).get('date'))
    scores['date'] = 1.0 if pred_date == gt_date and pred_date is not None else 0.0

    # tax_id (if present)
    pred_tax = pred.get('general_info', {}).get('tax_id')
    gt_tax = gt.get('general_info', {}).get('tax_id')
    if pred_tax is not None or gt_tax is not None:
        scores['tax_id'] = 1.0 if pred_tax == gt_tax else 0.0
    else:
        scores['tax_id'] = None  # Not present

    hard_score = sum(s for s in scores.values() if s is not None) / len([s for s in scores.values() if s is not None]) if any(s is not None for s in scores.values()) else 0.0
    return scores, hard_score

def evaluate_soft_fields(pred, gt):
    """Evaluate Soft Fields: Levenshtein similarity for seller_or_manufacturer, address."""
    scores = {}
    threshold = 0.85

    # seller_or_manufacturer
    pred_seller = pred.get('general_info', {}).get('seller_or_manufacturer', {}).get('value') if isinstance(pred.get('general_info', {}).get('seller_or_manufacturer'), dict) else pred.get('general_info', {}).get('seller_or_manufacturer')
    gt_seller = gt.get('general_info', {}).get('seller_or_manufacturer', {}).get('value') if isinstance(gt.get('general_info', {}).get('seller_or_manufacturer'), dict) else gt.get('general_info', {}).get('seller_or_manufacturer')
    sim_seller = levenshtein_similarity(pred_seller, gt_seller)
    scores['seller_or_manufacturer'] = 1.0 if sim_seller >= threshold else 0.0

    # address: use buyer_or_project as proxy for address
    pred_addr = pred.get('general_info', {}).get('buyer_or_project', {}).get('value') if isinstance(pred.get('general_info', {}).get('buyer_or_project'), dict) else pred.get('general_info', {}).get('buyer_or_project')
    gt_addr = gt.get('general_info', {}).get('buyer_or_project', {}).get('value') if isinstance(gt.get('general_info', {}).get('buyer_or_project'), dict) else gt.get('general_info', {}).get('buyer_or_project')
    sim_addr = levenshtein_similarity(pred_addr, gt_addr)
    scores['address'] = 1.0 if sim_addr >= threshold else 0.0

    soft_score = sum(scores.values()) / len(scores)
    return scores, soft_score

def evaluate_line_items(pred, gt):
    """Evaluate Line Items: F1-score based on matching items."""
    pred_items = pred.get('items', [])
    gt_items = gt.get('items', [])

    if not gt_items:
        return {}, 0.0

    # Match items by name similarity > 0.85
    matched_pred = set()
    matched_gt = set()
    threshold = 0.85

    for i, gt_item in enumerate(gt_items):
        gt_name = gt_item.get('name', {}).get('value') if isinstance(gt_item.get('name'), dict) else gt_item.get('name')
        for j, pred_item in enumerate(pred_items):
            if j in matched_pred:
                continue
            pred_name = pred_item.get('name', {}).get('value') if isinstance(pred_item.get('name'), dict) else pred_item.get('name')
            if levenshtein_similarity(gt_name, pred_name) >= threshold:
                # Check exact match for quantity, unit, unit_price, total_money
                gt_qty = normalize_number(gt_item.get('quantity', {}).get('value') if isinstance(gt_item.get('quantity'), dict) else gt_item.get('quantity'))
                pred_qty = normalize_number(pred_item.get('quantity', {}).get('value') if isinstance(pred_item.get('quantity'), dict) else pred_item.get('quantity'))
                gt_unit = gt_item.get('unit', {}).get('value') if isinstance(gt_item.get('unit'), dict) else gt_item.get('unit')
                pred_unit = pred_item.get('unit', {}).get('value') if isinstance(pred_item.get('unit'), dict) else pred_item.get('unit')
                gt_price = normalize_number(gt_item.get('unit_price', {}).get('value') if isinstance(gt_item.get('unit_price'), dict) else gt_item.get('unit_price'))
                pred_price = normalize_number(pred_item.get('unit_price', {}).get('value') if isinstance(pred_item.get('unit_price'), dict) else pred_item.get('unit_price'))
                gt_total = normalize_number(gt_item.get('total_money', {}).get('value') if isinstance(gt_item.get('total_money'), dict) else gt_item.get('total_money'))
                pred_total = normalize_number(pred_item.get('total_money', {}).get('value') if isinstance(pred_item.get('total_money'), dict) else pred_item.get('total_money'))

                if (gt_qty == pred_qty and gt_unit == pred_unit and gt_price == pred_price and gt_total == pred_total):
                    matched_pred.add(j)
                    matched_gt.add(i)
                    break

    tp = len(matched_gt)
    fp = len(pred_items) - tp
    fn = len(gt_items) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}, f1

def evaluate_invoice(pred_json, gt_json, weights=None):
    """Main evaluation function with weighted scoring."""
    if weights is None:
        weights = {'hard': 0.5, 'soft': 0.3, 'line': 0.2}  # Default weights

    with open(pred_json, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    hard_scores, hard_overall = evaluate_hard_fields(pred, gt)
    soft_scores, soft_overall = evaluate_soft_fields(pred, gt)
    line_scores, line_overall = evaluate_line_items(pred, gt)

    # Weighted combined metric
    weighted_score = (weights['hard'] * hard_overall +
                      weights['soft'] * soft_overall +
                      weights['line'] * line_overall)

    # STP: Straight-Through Processing - perfect if hard fields are 100%
    is_perfect = (hard_overall == 1.0)
    stp_percentage = 100.0 if is_perfect else 0.0  # For single invoice, it's 100% or 0%

    result = {
        'hard_fields': {'scores': hard_scores, 'overall': hard_overall},
        'soft_fields': {'scores': soft_scores, 'overall': soft_overall},
        'line_items': {'scores': line_scores, 'overall': line_overall},
        'weighted_score': weighted_score,
        'stp': {'is_perfect': is_perfect, 'percentage': stp_percentage}
    }

    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Path to predicted JSON')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON')
    args = parser.parse_args()

    result = evaluate_invoice(args.pred, args.gt)

    # Print weighted score as percentage
    weighted_percentage = result['weighted_score'] * 100
    print(f"Độ chính xác tổng quan (Weighted): {weighted_percentage:.1f}%")

    # Print STP
    stp_pct = result['stp']['percentage']
    print(f"Tỷ lệ hóa đơn hoàn hảo (STP): {stp_pct:.1f}%")

    # Print detailed field scores
    print("\n--- Chi tiết độ chính xác từng trường ---")
    hard_scores = result['hard_fields']['scores']
    for field, score in hard_scores.items():
        if score is not None:
            status = "Đạt" if score == 1.0 else "Không đạt"
            print(f"{field}: {status} ({score:.1f})")
        else:
            print(f"{field}: Không có dữ liệu")

    soft_scores = result['soft_fields']['scores']
    for field, score in soft_scores.items():
        status = "Đạt" if score == 1.0 else "Không đạt"
        print(f"{field}: {status} ({score:.1f})")

    line_overall = result['line_items']['overall']
    print(f"line_items: {'Đạt' if line_overall == 1.0 else 'Không đạt'} ({line_overall:.3f})")

    # Print full JSON result
    print("\n--- Kết quả đầy đủ ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))
