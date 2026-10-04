# Architectural and Experimental Decisions Log

| Decision ID | Date | Area | Decision Summary | Rationale |
|-------------|------|------|------------------|-----------|
| DEC-001 | 2026-10-01 | Environment | Target directory set to `scratch/ecte408` | Isolated project workspace complying with system guidelines |
| DEC-002 | 2026-10-01 | Compute | PyTorch CPU-optimized training pipeline | Snapdragon X 10-core CPU host with Adreno GPU; vectorized batching and float16/float32 caching to ensure high throughput |
| DEC-003 | 2026-10-01 | MATLAB | Verified MATLAB R2025a execution with required Image Processing & Statistics toolboxes | Meets exact ECTE408 requirements |
| DEC-004 | 2026-10-03 | Hyperparameters | Selected $D_0 = 0.20$ and NLM $h = 0.10$ | Equal-effort proxy search across $D_0 \in \{0.10, 0.15, 0.20, 0.30\}$ and $h \in \{0.05, 0.10, 0.15\}$ on validation data yielded highest degraded validation Dice at $D_0 = 0.20$ (0.0314 vs 0.0264–0.0311) and $h = 0.10$ |
