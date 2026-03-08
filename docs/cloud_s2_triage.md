# Cloud S2 Triage

- config_path: /workspace/deep-past-/configs/cloud_stage1_len512_lr2e4.yaml
- checkpoint_dir: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/best_model
- worth_continue_rerank: pending

## Baseline
- status: ok
- summary_path: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/diagnostics/val_diagnostic_summary_s1_final.json
- rowwise_csv: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/diagnostics/val_predictions_diagnostic_s1_final.csv
- reconstructed_csv: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/diagnostics/val_predictions_reconstructed_s1_final.csv
- reconstructed_geom: 9.3271
- reconstructed_bleu: 3.8132
- reconstructed_chrfpp: 22.8139
- health_empty_pct: 0.0000
- health_copy_pct: 0.0000
- health_shorter_half_pct: 0.3195
- health_pred_tok_p95: 2085.4000

## Decode Grid
- status: ok
- best_json: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/decode_grid_best.json
- metrics_csv: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/decode_grid_metrics.csv
- reconstructed_geom: 13.1616
- reconstructed_bleu: 6.5915
- reconstructed_chrfpp: 26.2808
- delta_geom_vs_baseline: +3.8346
- delta_bleu_vs_baseline: +2.7782
- delta_chrfpp_vs_baseline: +3.4669
- metric_level: parent_reconstructed
- health_empty_pct: NA
- health_copy_pct: NA
- health_shorter_half_pct: NA
- health_pred_tok_p95: NA

## Rerank N8
- status: pending
- summary_path: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/diagnostics/nbest_rerank_summary_stepBEST_beam8_n8_lp1p2_m512.json
- rowwise_csv: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/diagnostics/nbest_rerank_rowwise_stepBEST_beam8_n8_lp1p2_m512.csv
- candidates_csv: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/diagnostics/nbest_rerank_candidates_stepBEST_beam8_n8_lp1p2_m512.csv
- reconstructed_csv: /workspace/deep-past-/runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/diagnostics/nbest_rerank_reconstructed_stepBEST_beam8_n8_lp1p2_m512.csv
- reconstructed_geom: NA
- reconstructed_bleu: NA
- reconstructed_chrfpp: NA
- delta_geom_vs_baseline: NA
- delta_geom_vs_decode_grid: NA
- delta_bleu_vs_baseline: NA
- delta_chrfpp_vs_baseline: NA
- health_empty_pct: NA
- health_copy_pct: NA
- health_shorter_half_pct: NA
- health_pred_tok_p95: NA
