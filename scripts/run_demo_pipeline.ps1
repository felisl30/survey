$ErrorActionPreference = "Stop"

Write-Host "== S3 parse =="
python s3_model_code/parse_s3_outputs.py `
  --input-path outputs/s3/generation/flare_like_s3_raw_test_5_v4.csv `
  --output-path outputs/s3/generation/flare_like_s3_parsed_demo.csv

Write-Host "== S3 evaluate =="
python s3_model_code/evaluate_s3_answers.py `
  --input-path outputs/s3/generation/flare_like_s3_parsed_demo.csv `
  --output-path outputs/s3/evaluation/flare_like_s3_answer_results_demo.csv `
  --summary-path outputs/s3/evaluation/flare_like_s3_answer_summary_demo.json `
  --group-summary-path outputs/s3/evaluation/flare_like_s3_answer_summary_by_group_demo.csv

Write-Host "== S4 parse =="
python s4_model_code/parse_s4_outputs.py `
  --input-path outputs/s4/generation/fire_like_s4_raw_retrieve_test_5_hybrid_claims_llm_verify.csv `
  --output-path outputs/s4/generation/fire_like_s4_parsed_demo.csv

Write-Host "== S4 claim evaluation =="
python s4_model_code/evaluate_s4_claims.py `
  --input-path outputs/s4/generation/fire_like_s4_parsed_demo.csv `
  --claim-output-path outputs/s4/evaluation/fire_like_s4_claim_results_demo.csv `
  --summary-path outputs/s4/evaluation/fire_like_s4_claim_summary_demo.json `
  --group-summary-path outputs/s4/evaluation/fire_like_s4_claim_summary_by_group_demo.csv

Write-Host "== S4 answer evaluation =="
python s4_model_code/evaluate_s4_answers.py `
  --input-path outputs/s4/generation/fire_like_s4_parsed_demo.csv `
  --output-path outputs/s4/evaluation/fire_like_s4_answer_results_demo.csv `
  --summary-path outputs/s4/evaluation/fire_like_s4_answer_summary_demo.json `
  --group-summary-path outputs/s4/evaluation/fire_like_s4_answer_summary_by_group_demo.csv

Write-Host "== S4 final verification =="
python s4_model_code/verify_s4_results.py `
  --s4-parsed-path outputs/s4/generation/fire_like_s4_parsed_demo.csv `
  --claim-results-path outputs/s4/evaluation/fire_like_s4_claim_results_demo.csv `
  --answer-results-path outputs/s4/evaluation/fire_like_s4_answer_results_demo.csv `
  --output-dir outputs/s4/verification `
  --prefix demo

Write-Host "== Demo pipeline finished =="
