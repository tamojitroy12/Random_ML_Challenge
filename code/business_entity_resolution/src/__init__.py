"""Business entity resolution package (see pipeline.py / train_evaluate.py entry points).

Kept import-free on purpose: multiprocessing workers import src.records and should not pay
for LightGBM or other heavy imports.
"""
