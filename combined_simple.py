import os
import json
import argparse
from datetime import datetime
import re
from difflib import SequenceMatcher

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
    
    patterns = [
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  
    ]
    for pattern in patterns:
        match = re.match(pattern, date_str)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:  
                return f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
            else:  
                day, month, year = groups
                try:
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
    """Evaluate Hard Fields: exact match for total_payment, date, tax_id, code."""
    scores = {}
    pred_total = normalize_number(pred.get('financials', {}).get('total_payment', {}).get('value') if isinstance(pred.get('financials', {}).get('total_payment'), dict) else pred.get('financials', {}).get('total_payment'))
    gt_total = normalize_number(gt.get('financials', {}).get('total_payment', {}).get('value') if isinstance(gt.get('financials', {}).get('total_payment'), dict) else gt.get('financials', {}).get('total_payment'))
    scores['total_payment'] = 1.0 if pred_total == gt_total and pred_total is not None else 0.0

    pred_date = normalize_date(pred.get('general_info', {}).get('date', {}).get('value') if isinstance(pred.get('general_info', {}).get('date'), dict) else pred.get('general_info', {}).get('date'))
    gt_date = normalize_date(gt.get('general_info', {}).get('date', {}).get('value') if isinstance(gt.get('general_info', {}).get('date'), dict) else gt.get('general_info', {}).get('date'))
    scores['date'] = 1.0 if pred_date == gt_date and pred_date is not None else 0.0

    pred_tax = pred.get('general_info', {}).get('tax_id')
    gt_tax = gt.get('general_info', {}).get('tax_id')
    if pred_tax is not None or gt_tax is not None:
        scores['tax_id'] = 1.0 if pred_tax == gt_tax else 0.0
    else:
        scores['tax_id'] = None

    pred_code = pred.get('general_info', {}).get('code')
    gt_code = gt.get('general_info', {}).get('code')
    if pred_code is not None or gt_code is not None:
        scores['code'] = 1.0 if pred_code == gt_code else 0.0
    else:
        scores['code'] = None

    hard_score = sum(s for s in scores.values() if s is not None) / len([s for s in scores.values() if s is not None]) if any(s is not None for s in scores.values()) else 0.0
    return scores, hard_score

def evaluate_soft_fields(pred, gt):
    """Evaluate Soft Fields: Levenshtein similarity for seller_or_manufacturer, address."""
    scores = {}
    threshold = 0.85

    pred_seller = pred.get('general_info', {}).get('seller_or_manufacturer', {}).get('value') if isinstance(pred.get('general_info', {}).get('seller_or_manufacturer'), dict) else pred.get('general_info', {}).get('seller_or_manufacturer')
    gt_seller = gt.get('general_info', {}).get('seller_or_manufacturer', {}).get('value') if isinstance(gt.get('general_info', {}).get('seller_or_manufacturer'), dict) else gt.get('general_info', {}).get('seller_or_manufacturer')
    sim_seller = levenshtein_similarity(pred_seller, gt_seller)
    scores['seller_or_manufacturer'] = 1.0 if sim_seller >= threshold else 0.0

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
        weights = {'hard': 0.5, 'soft': 0.3, 'line': 0.2}

    with open(pred_json, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    hard_scores, hard_overall = evaluate_hard_fields(pred, gt)
    soft_scores, soft_overall = evaluate_soft_fields(pred, gt)
    line_scores, line_overall = evaluate_line_items(pred, gt)

    weighted_score = (weights['hard'] * hard_overall +
                      weights['soft'] * soft_overall +
                      weights['line'] * line_overall)

    valid_money = (hard_scores.get('total_payment', 0.0) == 1.0)
    valid_date = (hard_scores.get('date', 0.0) == 1.0)
    valid_seller = (soft_scores.get('seller_or_manufacturer', 0.0) >= 0.8)
    is_perfect = valid_money and valid_date and valid_seller
    stp_percentage = 100.0 if is_perfect else 0.0

    result = {
        'hard_fields': {'scores': hard_scores, 'overall': hard_overall},
        'soft_fields': {'scores': soft_scores, 'overall': soft_overall},
        'line_items': {'scores': line_scores, 'overall': line_overall},
        'weighted_score': weighted_score,
        'stp': {'is_perfect': is_perfect, 'percentage': stp_percentage}
    }

    return result

def run_batch_evaluation(pred_dir, easy_gt_dir, hard_gt_dir, output_file='combined_evaluation_results.json'):
    """Run batch evaluation for easy and hard datasets."""
    weights = {'hard': 0.5, 'soft': 0.3, 'line': 0.2}

    def evaluate_dataset(pred_dir, gt_dir, dataset_name):
        results = []
        all_weighted_scores = []
        all_hard_scores = []
        all_soft_scores = []
        all_line_scores = []
        perfect_invoices = 0
        total_invoices = 0

        gt_files = {os.path.splitext(f)[0]: os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.json')}

        for pred_filename in os.listdir(pred_dir):
            if not pred_filename.endswith('.json'):
                continue

            basename = os.path.splitext(pred_filename)[0]
            if basename not in gt_files:
                print(f"Warning: No ground truth for '{pred_filename}' in {dataset_name}")
                continue

            pred_path = os.path.join(pred_dir, pred_filename)
            gt_path = gt_files[basename]

            try:
                total_invoices += 1
                result = evaluate_invoice(pred_path, gt_path, weights)
                result['filename'] = basename
                results.append(result)

                all_weighted_scores.append(result['weighted_score'])
                all_hard_scores.append(result['hard_fields']['overall'])
                all_soft_scores.append(result['soft_fields']['overall'])
                all_line_scores.append(result['line_items']['overall'])
                if result['stp']['is_perfect']:
                    perfect_invoices += 1

            except Exception as e:
                print(f"Error processing {pred_filename} in {dataset_name}: {e}")

        if total_invoices == 0:
            return {'dataset': dataset_name, 'total_invoices': 0, 'error': 'No files evaluated.'}

        summary = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'total_invoices': total_invoices,
            'average_weighted_score': sum(all_weighted_scores) / total_invoices,
            'average_hard_score': sum(all_hard_scores) / total_invoices,
            'average_soft_score': sum(all_soft_scores) / total_invoices,
            'average_line_item_f1': sum(all_line_scores) / total_invoices,
            'stp_rate': (perfect_invoices / total_invoices) * 100,
        }

        return summary, results

    print("Evaluating easy dataset...")
    easy_results, easy_details = evaluate_dataset(pred_dir, easy_gt_dir, 'easy')

    print("Evaluating hard dataset...")
    hard_results, hard_details = evaluate_dataset(pred_dir, hard_gt_dir, 'hard')

    # Combine all results for overall hard and soft accuracies
    all_results = easy_details + hard_details
    total_invoices_overall = len(all_results)
    if total_invoices_overall > 0:
        overall_hard_avg = sum(r['hard_fields']['overall'] for r in all_results) / total_invoices_overall
        overall_soft_avg = sum(r['soft_fields']['overall'] for r in all_results) / total_invoices_overall
    else:
        overall_hard_avg = 0.0
        overall_soft_avg = 0.0

    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {'easy': easy_results, 'hard': hard_results},
        'overall_summary': {
            'total_invoices_easy': easy_results.get('total_invoices', 0),
            'total_invoices_hard': hard_results.get('total_invoices', 0),
            'total_invoices_overall': total_invoices_overall,
            'average_weighted_score_easy': easy_results.get('average_weighted_score', 0),
            'average_weighted_score_hard': hard_results.get('average_weighted_score', 0),
            'stp_rate_easy': easy_results.get('stp_rate', 0),
            'stp_rate_hard': hard_results.get('stp_rate', 0),
            'overall_hard_accuracy': overall_hard_avg,
            'overall_soft_accuracy': overall_soft_avg,
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to: {output_file}")
    print("\n--- OVERALL RESULTS ---")
    print(f"Easy: {easy_results.get('total_invoices', 0)} invoices, Avg Weighted: {easy_results.get('average_weighted_score', 0)*100:.2f}%, STP: {easy_results.get('stp_rate', 0):.2f}%")
    print(f"Hard: {hard_results.get('total_invoices', 0)} invoices, Avg Weighted: {hard_results.get('average_weighted_score', 0)*100:.2f}%, STP: {hard_results.get('stp_rate', 0):.2f}%")
    print(f"Overall: {total_invoices_overall} invoices, Hard Accuracy: {overall_hard_avg*100:.2f}%, Soft Accuracy: {overall_soft_avg*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate OCR model for Vietnamese invoices.")
    subparsers = parser.add_subparsers(dest='mode', help='Choose evaluation mode')

    # Single invoice mode
    single_parser = subparsers.add_parser('single', help='Evaluate a single invoice')
    single_parser.add_argument('--pred', required=True, help='Path to predicted JSON')
    single_parser.add_argument('--gt', required=True, help='Path to ground truth JSON')
    single_parser.add_argument('--output', help='Output file for single invoice result (optional)')

    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Evaluate multiple invoices')
    batch_parser.add_argument('--pred_dir', required=True, help='Directory with predicted JSONs')
    batch_parser.add_argument('--easy_gt_dir', required=True, help='Directory with easy ground truth JSONs')
    batch_parser.add_argument('--hard_gt_dir', required=True, help='Directory with hard ground truth JSONs')
    batch_parser.add_argument('--output_file', default='combined_evaluation_results.json', help='Output file for results')

    args = parser.parse_args()

    if args.mode == 'single':
        result = evaluate_invoice(args.pred, args.gt)
        weighted_percentage = result['weighted_score'] * 100
        stp_pct = result['stp']['percentage']
        print(f"Weighted Accuracy: {weighted_percentage:.1f}%")
        print(f"STP: {stp_pct:.1f}%")
        print("Detailed results:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    elif args.mode == 'batch':
        run_batch_evaluation(args.pred_dir, args.easy_gt_dir, args.hard_gt_dir, args.output_file)
    else:
        parser.print_help()
