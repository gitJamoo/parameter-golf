# Universal Transformer with Depth Embeddings

**Track:** Non-record, unlimited compute (4-hour)
**Architecture:** True Universal Transformer — single shared block + learnable depth embeddings
**Target:** Demonstrate the "Universal Transformer" requested in the challenge README

---

## What This Is

The challenge README specifically requests a Universal Transformer on the 4-hour unlimited track:
> "Universal transformer — We have lots of depth recurrence submissions, but I'd love to see one 4 hour"

This submission implements a **true** Universal Transformer (Dehghani et al., 2019), which is distinct from the depth-recurrent Middle-Cycle submissions (PRs #325, #363, etc.) in two key ways:

1. **Single shared block** — all recurrence steps use exactly one transformer block, not a cycle of N blocks. The shared block is `shared_block`, not `blocks[i % N]`.
2. **Depth embeddings** — a learnable `(num_recurrence × model_dim)` embedding table is indexed by step number. `depth_embeddings[step]` is added to the residual stream before each application of the shared block, giving the block a learned signal about which step it is on. Previous recurrence submissions used DepthLoRA adapters; this is the original UT mechanism.

## Architecture

```
input_ids
    │
    ▼
tok_emb  ──►  F.rms_norm  ──►  x0 (anchor)
                                    │
    ┌───────────────────────────────┘
    ▼
stem_blocks[0]  ──►  stem_blocks[1]  ──► ...  ──► stem_blocks[num_stem-1]
    │                      │
   skip₀               skip₁  (stored for U-Net)
                                    │
    ┌───────────────────────────────┘
    │
    ▼   ┌─────────────────────────────────────────────────────────────┐
    │   │  for step in range(num_recurrence):                         │
    │   │      x = x + depth_embeddings[step]  ◄── step signal       │
    └──►│      x = shared_block(x, x0)          ◄── same weights      │
        │  (24 iterations, one block)                                 │
        └─────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┘
    ▼
+ skip₁ × skip_weights[0]
    │
tail_blocks[0]
    │
+ skip₀ × skip_weights[1]
    │
tail_blocks[1]
    │
    ▼
final_norm  ──►  lm_head  ──►  softcap  ──►  cross_entropy
```

## Key Design Choices

### Depth Embeddings (UT-original)
The original Universal Transformer paper adds a depth-wise positional encoding to the residual stream at each step. Here we use a learnable version initialized near zero with `std=0.01`. The block sees: `x + depth_embeddings[step]`, giving it step-level context without separate weights per step.

This is different from DepthLoRA (PR #363), which adds a rank-r adapter to Q and V projections. Depth embeddings affect the entire residual stream rather than just attention queries/values.

### Noisy QAT on the Shared Block
As established in PR #363, quantization error compounds through recurrence steps: the same weights are quantized once at export but the error is amplified ~N times through N repeat applications. We address this with int8-calibrated uniform noise injected into the shared block's weights during training:

```python
amax = w.detach().float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
step_size = (amax / 127.0).to(w.dtype)
w = w + (torch.rand_like(w) - 0.5) * step_size
```

This trains the model to be robust to quantization-scale perturbations. PR #363 reported a collapse from 0.37 bpb quantization gap to 0.002 bpb with this technique (for int8 targets).

### Wider Model (640d vs 512d)
With 5 unique stored blocks instead of 9–11, the freed parameter budget is invested in model width. At 640d with mlp_mult=2:
- ~15M stored parameters → ~13MB artifact (well under 16MB)
- 28 effective depth (2+24+2) vs 9 in the baseline

### 4-Hour Budget
The previous depth recurrence analysis (PR #363) identified two structural taxes:
1. **Quantization compounding** → addressed by Noisy QAT
2. **Step-time overhead** → ~32ms/step slower, causing ~22% fewer training steps in 10 minutes

At 4 hours, the step-time tax matters far less. With ~200ms/step (estimated for 28-deep 640d model), we get ~72,000 training steps — vastly more than the baseline's 20,000 steps in 10 minutes. This allows the parameter efficiency of weight-sharing to actually manifest.

## Parameter Budget

| Component | Params |
|-----------|--------|
| tok_emb (1024 × 640) | 655,360 |
| stem_blocks[0] (640d, mlp×2) | ~2,867,200 |
| stem_blocks[1] | ~2,867,200 |
| shared_block | ~2,867,200 |
| tail_blocks[0] | ~2,867,200 |
| tail_blocks[1] | ~2,867,200 |
| depth_embeddings (24 × 640) | 15,360 |
| skip_weights (2 × 640) | 1,280 |
| norms, gains, etc. | ~5,000 |
| **Total** | **~15,013,000** |

At int8 + zlib: ~13MB artifact + ~50KB code ≈ **~13MB total** (well under 16MB).

## Differences from Prior Depth Recurrence Work

| | Middle-Cycle (PR #325, #363) | This submission |
|--|--|--|
| Shared blocks | N unique blocks cycled as `blocks[i % N]` | 1 truly shared block |
| Step signal | DepthLoRA: rank-r Q/V adapters per step | Depth embeddings: full residual offset per step |
| Quantization | STE QAT / none | Noisy QAT calibrated for int8 on shared block |
| Training | 10-min track | 4-hour unlimited track |
| Configuration | 3x3 / 2x5 | 2+24+2 (stem+shared×R+tail) |

## Running

```bash
# 8xH100, 4-hour unlimited track
RUN_ID=universal_transformer_4h \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=14400 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key env vars to sweep:
- `NUM_RECURRENCE` (default 24) — how many times to apply the shared block
- `NUM_STEM` (default 2) / `NUM_TAIL` (default 2) — unique pre/post blocks
- `MODEL_DIM` (default 640) — model width
- `NOISY_QAT_SHARED` (default 1) — toggle Noisy QAT on shared block
- `MATRIX_LR` (default 0.04) — Muon learning rate for block weight matrices

## Expected Behavior

The model initially trains with all recurrence steps identical (depth_embeddings ≈ 0, so the block sees the same input each step). As training progresses:
1. Gradients from the loss will push different steps to specialize
2. depth_embeddings will grow to encode step-level information
3. The shared block learns to execute different "programs" based on the depth embedding it receives

At initialization, the shared block's `proj` layers are zero-initialized, making each application a near-identity transform. Training should be numerically stable.

## Relationship to Prior Art

- **Dehghani et al. 2019** — original Universal Transformer with sinusoidal depth embeddings and ACT
- **Huginn (Alibaba 2025)** — scaled UT for inference-time compute scaling
- **PR #325 (Aum08Desai)** — Middle-Cycle architecture that inspired the 3x3 configuration
- **PR #363 (Kamin)** — definitive analysis of why recurrence fails on 10-min track; Noisy QAT contribution

The ACT (Adaptive Computation Time) mechanism from the original UT paper is not implemented here — it requires dynamic per-token halting that's incompatible with `torch.compile(fullgraph=True)`. A static recurrence count is used instead.
