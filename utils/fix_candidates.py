"""
Quick script to deduplicate candidate_pairs.tsv and run validation.
"""
import polars as pl
import subprocess
import sys

print("Deduplicating output/candidate_pairs.tsv...")
cand_path = "output/candidate_pairs.tsv"
df = pl.read_csv(cand_path, separator="\t")
print(f"Initial rows: {len(df):,}")

df_unique = df.unique(subset=["source1_entity_id"], keep="last")
print(f"Unique rows:  {len(df_unique):,}")

df_unique.write_csv(cand_path, separator="\t")
print("Saved clean output/candidate_pairs.tsv!")

print("\nRunning submission validator...")
cmd = [
    sys.executable,
    "utils/validate_submission.py",
    "--matching", "output/matching_results.tsv",
    "--candidate", "output/candidate_pairs.tsv",
    "--test-dir", "dataset/test"
]
subprocess.run(cmd)
