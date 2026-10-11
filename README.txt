H20 C1/C2 compression/cache fusion evidence — vLLM PR #61088

Latest: dual-tp4/REPORT.txt and dual-tp4/analysis-summary.json record the complete two-TP4 GPU/NUMA crossover, all 72 measured batches and all performance directions. Cold 16K/c4 shows a positive signal in both placements; high-concurrency results do not show a gain, and other small-batch cells change direction. Stable general end-to-end acceleration is not established. The PR remains Draft.

The earlier single-engine sequential cohort is retained under model/. It is not pooled with the crossover. Isolated operator results and tests are under integration/; initial evidence files remain unchanged.

The full new raw archives remain on the test host with SHA/size records. This branch publishes aggregate samples and exact benchmark/analysis scripts, not those full archives. See the reports for real DSpark acceptance, observed cache-hit ratios, finite-batch limitations and pending NCU/human review.
