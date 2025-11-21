# TODO for Cleaning up run.py

- [x] Remove old functions: parse_texts, parse_body_to_columns, predict_with_structure, parse_table_html_to_json
- [x] Update extract_invoice_fields: Modify prompt to extract total_amount and default quantity to 1 if not clear
- [x] Simplify main block: Remove PPStructure logic, keep only OCR prediction, sort texts, extract fields, save cleaned JSON
- [x] Verify the changes by running the script (if possible)
