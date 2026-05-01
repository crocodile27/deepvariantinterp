# Population Complexity Projection — Findings

*Generated from `scripts/population_complexity.py`*

---

## 1. Sample Inventory

**Included samples (n=13, including NA12878):**

| Sample | Population | Superpop | n_sites | Mean score |
|--------|-----------|----------|---------|------------|
| HG01985 | LWK | AFR | 427 | 0.316 |
| HG02922 | LWK | AFR | 381 | 0.034 |
| HG01048 | CLM | AMR | 390 | 0.731 |
| HG01197 | CLM | AMR | 336 | -1.160 |
| HG01565 | PEL | AMR | 380 | -0.559 |
| HG00759 | CHS | EAS | 384 | -2.332 |
| NA18939 | JPT | EAS | 359 | -0.380 |
| HG00864 | CDX | EAS | 348 | -1.227 |
| NA12878 | CEU | EUR | 337 | -0.000 |
| HG00731 | GBR | EUR | 307 | -0.660 |
| NA20502 | TSI | EUR | 373 | 0.346 |
| HG03009 | GIH | SAS | 329 | -0.123 |
| NA20847 | GIH | SAS | 412 | 0.062 |

**Excluded:** NA12878 is included in the EUR reference distribution but was the probe training sample (noted in caveats).
No samples from the 13-sample active set were missing — all 12 non-NA12878 samples had usable embeddings.

---

## 2. Kruskal-Wallis Test

**H = 12.671, p = 1.3000e-02**

There IS a statistically significant difference in complexity scores across superpopulations (p < 0.05).

Sites per superpopulation:
- AFR: 808 sites
- AMR: 1106 sites
- EAS: 1091 sites
- EUR: 1017 sites
- SAS: 741 sites

---

## 3. Pairwise Results (vs EUR, Bonferroni-corrected)

- **AFR**: d=0.029 (higher, small effect), p_bonf=1.000e+00
- **AMR**: d=-0.023 (lower, small effect), p_bonf=1.000e+00
- **EAS**: d=-0.135 (lower, small effect), p_bonf=2.007e-02
- **SAS**: d=0.006 (higher, small effect), p_bonf=1.000e+00

**Significant differences after Bonferroni correction:** EAS

---

## 4. Direction of Effect

EUR mean complexity score: -0.072 ± 9.015 (includes NA12878, the probe training sample).

Per-superpopulation consistency across samples:
- **EUR**: sample means = [-0.000, -0.660, 0.346] (mixed directions)
- **AFR**: sample means = [0.316, 0.034] (consistent direction)
- **AMR**: sample means = [0.731, -1.160, -0.559] (mixed directions)
- **EAS**: sample means = [-2.332, -0.380, -1.227] (consistent direction)
- **SAS**: sample means = [-0.123, 0.062] (mixed directions)

Non-EUR sites trend **lower** (more complex / uncertain) than EUR sites on average.

---

## 5. Caveats

1. **EUR circular reference bias:** The probe was trained on NA12878 (EUR/CEU). EUR is simultaneously the probe training reference and a comparison group. This creates a structural advantage for EUR sites scoring as TP-like. The effect sizes for non-EUR populations must be interpreted with this in mind.

2. **Small sample sizes:** n=2–3 samples per superpopulation, covering a 100 kb region of chr20. Effects should be replicated on full-chr20 data before drawing strong conclusions.

3. **GIAB benchmark coverage confound:** The complexity axis captures GIAB callability (high-confidence vs uncertain regions), not purely genomic complexity. Some EUR advantage may reflect GIAB benchmark coverage bias — the GIAB high-confidence regions were originally defined on EUR samples — rather than model bias per se.

4. **Single region:** All data comes from chr20:10,000,000–10,100,000 (100 kb). Population-specific structural variants or coverage patterns in this window could drive apparent effects.

---

## 6. Next Steps

- **Full chr20 analysis:** Extend embeddings to the full chr20 (≈50 Mb). This would increase site counts from ~300–400 to tens of thousands per sample, providing statistical power to detect small effect sizes (d < 0.2).

- **GIAB truth sets for non-EUR samples:** If GIAB truth sets exist for AFR/AMR/EAS/SAS samples (e.g., HG002–HG007), rerun the linear probe analysis on those samples to get a ground-truth calibration that is independent of EUR.

- **Channel attribution:** Use the Cohen's d channel analysis (already run for NA12878) to identify which mixed5 channels drive the population complexity differences. If population-informative channels overlap with complexity-informative channels, this is direct evidence of entanglement.

- **GRCh38 cross-reference:** Compare complexity scores for the same samples against GRCh38 to disentangle reference-build effects from population effects.
