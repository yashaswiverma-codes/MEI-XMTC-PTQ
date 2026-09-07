# MEI-XMTC-PTQ

**Memory-Efficient Inferencing in Extreme Multi-Label Text Classification Using Post-Training Quantization**

Official code release accompanying the paper (IEEE DSAA 2026, Short Presentation).

This repository implements Post-Training Quantization (PTQ) for two Extreme Multi-Label Classification (XMTC) methods — **DiSMEC** and **AnnexML** — evaluated across standard XMTC benchmark datasets.

## Repository Structure

```
├── dismecpp/                          # DiSMEC PTQ implementation
│   ├── src/                           # Source code (including bundled cnpy library)
│   ├── data/eurlex/                   # Sample dataset (EURLex-4K only — see Datasets below)
│   ├── test/                          # Test fixtures
│   └── *.model                        # Small baseline model metadata files
├── annexml/                           # AnnexML PTQ implementation
├── annexml_quantization_scripts/      # AnnexML quantization helper scripts
├── dismec_quantization_scripts/       # DiSMEC quantization helper scripts
├── run_dismec_patha.sh                # DiSMEC Path A (weight-only quantization) pipeline
├── run_dismec_pathb.sh                # DiSMEC Path B (activation-quantized) pipeline
├── run_dismec_larger_dataset_inference_timing.sh
├── run_annexml_patha.sh               # AnnexML Path A pipeline
└── run_annexml_pathb.sh               # AnnexML Path B pipeline
```

## Quantization Paths

Two inference modes are implemented for both DiSMEC++ and AnnexML:

- **Path A (weight-only quantization, default):** Weights are quantized (INT8/INT4); activations remain float32. Weights are dequantized before scoring.
- **Path B (`--act_quant`):** Both weights and activations are quantized to INT8. Scoring uses integer dot products, rescaled by weight and activation scales.

18 quantization configurations are benchmarked per dataset, spanning row/group, symmetric/asymmetric, mixed-precision, and activation-quantized variants.

## Datasets

Benchmark datasets are from the **Extreme Classification Repository**: http://manikvarma.org/downloads/XC/XMLRepository.html

| Dataset        | Included in repo? |
|----------------|--------------------|
| EURLex-4K      | Yes (`dismecpp/data/eurlex/`) |
| Wiki10-31K     | No — download separately |
| AmazonCat-13K  | No — download separately |
| Amazon-670K    | No — download separately |
| Delicious-200K | No — download separately |

Only EURLex-4K is bundled as a lightweight reproducibility sample; the remaining datasets exceed GitHub's file-size limits. Download and place them under the matching `data/<dataset>/` folder to reproduce full results.

Trained model checkpoints (`.model.weights-*` files) are not included due to size (multi-GB per model) and can be regenerated via the training scripts, or requested directly from the authors.

## Dependencies

- **CLI11** (command-line parsing) — https://github.com/CLIUtils/CLI11
- **Eigen** (linear algebra) — https://eigen.tuxfamily.org/
- **cnpy** (NumPy .npz I/O in C++, MIT license) — https://github.com/rogersce/cnpy — bundled directly in `dismecpp/src/cnpy/`
- OpenBLAS, OpenMP, AVX2-capable CPU (for SIMD kernels)

## Build

```bash
cd dismecpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

(AnnexML build instructions — adjust per your build system, e.g. Makefile or CMake)

## Running the Pipeline

```bash
# DiSMEC — Path A (weight-only) on EURLex
bash run_dismec_patha.sh eurlex

# DiSMEC — Path B (activation-quantized) on EURLex
bash run_dismec_pathb.sh eurlex

# AnnexML — Path A / Path B
bash run_annexml_patha.sh
bash run_annexml_pathb.sh
```

Each script quantizes the baseline model across all configurations, runs inference, evaluates against standard XMC metrics (P@k, nDCG@k, PSP@k, PSnDCG@k), and writes a summary CSV to the corresponding `results/<dataset>/` folder.

## Citation

If you use this code, please cite:

```
[BibTeX entry — add once available]
```

## License

MIT License — see [LICENSE](LICENSE).
