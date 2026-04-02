@echo off
set RUN_ID=universal_transformer_4h
set DATA_PATH=./data/datasets/fineweb10B_sp1024/
set TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
set VOCAB_SIZE=1024
set MAX_WALLCLOCK_SECONDS=14400
torchrun --standalone --nproc_per_node=8 records/track_non_record_16mb/2026-03-28_UniversalTransformer/train_gpt.py
