# M2-T10: figure notes for the M2-T05 report

Segmentation: **High spoilage risk** = batches whose time needed to sell the stock is longer than the days left before expiry (`sell_through_ratio` above 1, or zero velocity with stock). **Normal velocity** = all other batches with a velocity. 576 batches with no earlier sales data are excluded.

| Figure | File | Suggested caption |
|---|---|---|
| 1 | `reports/figures/m2/M2_T10_01_segment_overview.png` | 58.9% of batches are high risk, but both segments spoil at about the same rate (19.5% and 19.3%). The high-risk segment loses less per batch (21.3% against 24.4% waste). |
| 2 | `reports/figures/m2/M2_T10_02_ratio_bands.png` | Spoilage stays between 19.0% and 20.0% in every sell-through ratio band. |
| 3 | `reports/figures/m2/M2_T10_03_category_segments.png` | The high-risk share runs from 47.6% (Pharmaceuticals) to 67.0% (Frozen_Meals). |
| 4 | `reports/figures/m2/M2_T10_04_days_to_sell_vs_expiry.png` | For a typical batch the time needed to sell the stock is close to the time left, so the ratio works better as a graded score than as a yes/no flag. |
| 5 | `reports/figures/m2/M2_T10_05_quality_grade_by_segment.png` | Quality grade separates spoilage in both segments (grade A 15.6% and 14.7%, grade C 25.2% and 26.9%). |
| 6 | `reports/figures/m2/M2_T10_06_discounts_by_segment.png` | High-risk batches are discounted more often (37.2% against 21.2%) and more deeply (9.1% against 2.6% deep discounts). |
| 7 | `reports/figures/m2/M2_T10_07_dashboard.png` | One-page summary of the segmentation. |

Tables: `reports/tables/M2_T10_segment_summary.csv`, `M2_T10_category_segments.csv`, `M2_T10_ratio_bands.csv`, `M2_T10_grade_by_segment.csv`.
