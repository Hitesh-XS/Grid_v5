### Task A (Primary): Congestion Severity Classifier
Predicts four categorical severity tiers: `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. Powered by a soft-voting ensemble comprising **CatBoost (40%)**, **LightGBM (30%)**, and **XGBoost (30%)**.

### Task B: Resource Allocation Engine
Predicts structural response configurations instantly upon incident registration:
* `officers_needed` (Integer)
* `barricades_needed` (Integer)
* `response_priority` (`Low`, `Medium`, `High`, `Immediate`)

### Task C: Future Hotspot Prediction
Regressive model predicting which specific corridors or junctions are highly likely to generate an autonomous event within the next **2 to 6 hours**.

### Task D: Unplanned Event Risk Analytics
Calculates the spatial-temporal anomaly threshold to determine the probability that a specific point coordinates pair defaults into a vehicle breakdown, accident, or road blockage.

---

## 🛠️ Data Pipeline & Engineering Blueprint

### Phase 2 & 3: Audit, Cleanse & Smart Imputation
* **Weak/Useless Features Extracted:** Structural fields (`id`, `created_by_id`, `client_id`, `kgid`, `vehicle_no`) and highly sparse unstructured variables (`comment`, `meta_data`, `map_file` with $>95\%$ missing rates) are filtered out immediately.
* **Spatial KNN Imputation:** Missing spatial indices (`zone`, `junction`) are not blindly replaced with static `"Unknown"` indicators. Instead, missing entries are derived via non-parametric `KNNImputer` mapping over `latitude` and `longitude` arrays.
* **Coordinate Mirroring:** Missing terminal markers are resolved dynamically:
    $$\text{endlatitude} = \text{latitude}$$
    $$\text{endlongitude} = \text{longitude}$$
* **Unresolved Temporal Tracking:** Event durations are computed via:
    $$\Delta t = t_{\text{modified}} - t_{\text{start}}$$
    Unresolved incident durations are conditionally filled using the median historical duration partitioned strictly by `event_type`.

### Phase 4 & 5: Spatial Intelligence & Temporal Cycles
* **HDBSCAN Clustering:** Standard K-Means is explicitly rejected due to its spherical cluster assumptions. Densities are mapped using **HDBSCAN** via a haversine distance metric to extract asymmetric real-world bottlenecks (e.g., *Hebbal*, *Silk Board*, *Peenya*), outputting a localized `hotspot_id`.
* **Rush Hour Extraction:** Explicit binary signaling flags are raised based on explicit operational boundaries:
    $$\text{rush\_hour} = \mathbb{I}(7 \le \text{hour} \le 11 \lor 16 \le \text{hour} \le 21)$$
* **Cyclic Encoding:** To preserve continuous temporal relationships (e.g., ensuring hour 23 maps closely to hour 0), raw hour sequences undergo non-linear trigonometric projection:
    $$\text{hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

### Phase 6 & 7: Feature Aggrandizement & Target Formulation
* **Lagged Historical Aggregates:** Calculates running frequency counts over critical windows (`past_1_day_events`, `past_7_day_events`, `past_30_day_events`) grouped by corridors, junctions, and zones.
* **Synthesized Congestion Scoring:** In the absence of direct loop-detector telemetry (speed/flow), a continuous multi-criteria target index is computed:
    $$\text{Severity Score} = \left(0.30 \cdot P + 0.25 \cdot C + 0.20 \cdot E_{cause} + 0.15 \cdot D_{norm} + 0.10 \cdot L_{risk}\right) \times 100$$
    Where $P$ represents priority weight, $C$ signifies road closure impact, $E_{cause}$ models risk variance by incident origin, $D_{norm}$ represents scaled duration, and $L_{risk}$ models spatial cluster frequency.
* **Target Binning Matrix:**
    * `[00 - 25)` $\rightarrow$ **LOW**
    * `[25 - 50)` $\rightarrow$ **MODERATE**
    * `[50 - 75)` $\rightarrow$ **HIGH**
    * `[75 - 100]` $\rightarrow$ **CRITICAL**

---

## 🚀 Training Engine & Validation Strategy

### Time-Series Cross-Validation (Phase 9)
Standard K-Fold cross-validation introduces fatal data leakage when training over historical traffic flows. GridV1 strictly enforces an expanding window `TimeSeriesSplit` mechanism. Models evaluating April events are exclusively trained on data spanning January through March.
