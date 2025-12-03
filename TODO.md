# TODO for Invoice Extraction Metric Implementation

- [ ] Create evaluate_metrics.py with normalization functions (remove commas, standardize dates)
- [ ] Implement exact match function for Hard Fields (total_payment, date, tax_id)
- [ ] Implement Levenshtein similarity function for Soft Fields (seller_or_manufacturer, address)
- [ ] Implement F1-score function for Line Items (match by name similarity, then exact match on fields)
- [ ] Create main evaluation function to compute scores for each group and combined metric
- [ ] Test the metric on sample ground truth and predicted JSONs
- [ ] Integrate into evaluation pipeline if needed
