# Cloud Next Steps

1. S3 first: scale the GC short-alignment pool beyond 10k rows by gradually relaxing `max_align_cost`, while keeping `--no-allow-replacement`.
2. Compare the S3 pool reports at cost `1.4 / 1.6 / 1.8`, and keep the strongest pool that still preserves alignment quality.
3. S4 next: run stage2 curriculum from the strong stage1 checkpoint `runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/best_model`.
4. Do not auto-start S3 or S4 from this note. The prepared commands live in:
   - `scripts/run_s3_gc_pool_scale.sh`
   - `scripts/run_s4_stage2_curriculum.sh`
