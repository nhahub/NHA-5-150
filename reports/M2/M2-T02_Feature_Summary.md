# M2-T02: Remaining shelf-life ratio and sales velocity

*Text for the Feature Engineering Summary. Evidence: `M2-T02.ipynb` (run on the verified file, SHA-256 `de94302b...`).*

## Features

| Feature | Formula | Columns used | Notes |
|---|---|---|---|
| `remaining_shelf_life_ratio` | `days_until_expiry / shelf_life_days`, clipped to [0, 1]; empty if `shelf_life_days` <= 0 | `days_until_expiry`, `shelf_life_days` | Near 0: close to or past expected life. Near 1: fresh. Puts products with different shelf lives on one scale. Equals `expiry_remaining_ratio` from M1-T04. |
| `remaining_ratio_at_purchase` (extra) | `days_remaining_at_purchase / shelf_life_days`, clipped to [0, 1] | `days_remaining_at_purchase`, `shelf_life_days` | Same idea with the column that equals the date gap exactly. |
| `sales_velocity` | sum of `units_sold` of earlier batches of the same `product_name` in the same store in the 7 days before the batch, divided by 7 | `units_sold`, `product_name`, `store_id`, `transaction_date` | Earlier days only, so the batch's own `units_sold` is excluded. |
| `sales_velocity` (fallback) | units sold in earlier days in the same category and store, divided by the days since the first record of that group (at least 7) | `units_sold`, `category`, `store_id`, `transaction_date` | Used when the pair has less than 7 days of history or no earlier batch in the window (74.0% of rows). |
| `sales_velocity_per_day` (extra) | average of `units_sold / days_remaining_at_purchase` over the same earlier batches, with the same fallback | `units_sold`, `days_remaining_at_purchase`, ... | Real units per day. Use it for stock / velocity calculations. |
| `velocity_source` | `product_store_7d` (25.4%), `category_store_avg` (74.0%), `no_history` (0.6%) | | Shows which rule produced the value. `no_history` rows (the first days of the file) have an empty velocity. |
| `batches_in_window_7d` | number of earlier batches in the 7-day window | `product_name`, `store_id`, `transaction_date` | How many batches the value is based on. |

## Decisions and reasons

1. **The product-store pair is `product_name` x `store_id`, not `product_id` x `store_id`.** `product_id` has 45,301 values and gives 97,984 pairs with 1.02 rows each (98.0% of pairs have a single row, so no history exists). `product_name` gives 3,150 pairs with about 32 rows each.
2. **An empty 7-day window counts as "not enough history" and uses the fallback.** 73.8% of the pair windows contain no earlier batch. Reading them as velocity 0 would make three quarters of the feature say that the product does not sell.
3. **The window stays at 7 days (card).** Widening to 14 or 28 days still leaves 54.6% and 30.6% of the windows empty.
4. **The official ratio uses `days_until_expiry` (card).** It is 0 to 3 days below the date gap in 74,191 rows. The lag is unrelated to `was_spoiled` (74.3% of not spoiled and 73.6% of spoiled batches have a lag) and to distribution hours, handling, temperature abuse events and supplier score (correlations within 0.004), so there is no sign of leakage.
5. **An extra velocity in units per day is added.** `units_sold` is a batch's total over its whole life (`units_sold + units_wasted = initial_quantity` in every row), so the card's velocity mixes time scales (frozen meals: 137 units over about 212 days; ready-to-eat: 220 units in 3 days). `units_sold / days_remaining_at_purchase` correlates 0.918 with `daily_demand` at batch level; raw `units_sold` only 0.44.

## Validation

- 300-row recalculation with a plain loop for the product-store rule and 300 rows for the fallback rule: 0 mismatches for `sales_velocity`, `sales_velocity_per_day` and `batches_in_window_7d`.
- Leakage test: setting the own `units_sold` of one batch per category-store group to 1,000,000 changes none of those batches' own velocities (0 of 500 changed) and changes 35,561 later batches, as it should.
- Ratio checks: no missing values, range exactly 0 to 1, equal to the card's long formula, 5 rows verified by hand.
- Row checks: 100,000 rows in and out, `record_id` unique, no negative velocities.

## Limitations (measured)

- `units_sold` is a batch's final total, so on average 78.3% of the sales in a window come from batches that had not finished selling on day `t`. Inside a category, the velocity of earlier batches has almost no relation to the current batch (correlation 0.007 for the per-day velocity, 0.002 for the card's velocity), so no information about the current batch passes through them and the feature is not distorted in this dataset. In deployment, count only finished batches.
- Neither velocity predicts spoilage on its own (AUC 0.503 and 0.513). Both ratios are weak (AUC 0.481 and 0.474). The velocities are mostly a category-level signal; their value is for the sell-through ratio and the revenue and waste side of Milestone 3.
