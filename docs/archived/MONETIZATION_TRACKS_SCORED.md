# Monetization tracks: scored and ranked

Scoring and ranking of monetization options for the RecGPT/elixir-recgpt stack. Scores are 1–5 (5 = best). **Judge** = weighted overall score used for ranking.

---

## Scoring criteria (1–5)


| Criterion              | Meaning                                                    |
| ---------------------- | ---------------------------------------------------------- |
| **Revenue potential**  | Ceiling: how much money this track could generate at scale |
| **Time to revenue**    | How quickly you can get first paying use (weeks/months)    |
| **Fit with codebase**  | How much already exists; how little net new work           |
| **Differentiation**    | How defensible vs. generic recsys APIs / consultants       |
| **Sales friction**     | Ease of finding buyers, contracts, and pricing clarity     |
| **Operational burden** | Ongoing cost to run, support, and maintain                 |


---

## Track scores


| Track                                        | Revenue potential | Time to revenue | Fit with codebase | Differentiation | Sales friction | Operational burden | **Judge** |
| -------------------------------------------- | ----------------- | --------------- | ----------------- | --------------- | -------------- | ------------------ | --------- |
| 1. Recommendation API / SaaS                 | 5                 | 5               | 5                 | 2               | 4              | 3                  | **4.2**   |
| 2. Gaming & digital storefronts              | 4                 | 4               | 5                 | 3               | 3              | 3                  | **3.8**   |
| 3. Multimodal catalog / “recommend by image” | 4                 | 3               | 4                 | 5               | 3              | 3                  | **3.8**   |
| 4. Prediction markets & trading-adjacent     | 5                 | 2               | 3                 | 4               | 2              | 2                  | **3.2**   |
| 5. Library / platform licensing              | 3                 | 4               | 5                 | 2               | 4              | 4                  | **3.7**   |
| 6. Consulting & custom pipelines             | 4                 | 3               | 5                 | 2               | 3              | 2                  | **3.4**   |
| 7. Low-latency / edge positioning            | 3                 | 3               | 4                 | 3               | 3              | 3                  | **3.2**   |


---

## Judge formula and per-track notes

**Judge** = (Revenue × 0.25) + (Time to revenue × 0.20) + (Fit × 0.20) + (Differentiation × 0.15) + (Sales friction × 0.10) + (Operational burden × 0.10).  
Rounded to one decimal.

- **1. Recommendation API / SaaS** — Highest judge (4.2). Strong revenue ceiling, fast to ship (gRPC + serve already there), best codebase fit. Differentiation is “good recsys” not “unique IP”; sales and ops are manageable.
- **2. Gaming & digital storefronts** — 3.8. Good revenue and time-to-revenue; codebase and Steam/Figgie work fit well. Some differentiation (games + assets); sales require vertical relationships.
- **3. Multimodal catalog / “recommend by image”** — 3.8. Strong differentiation (vision + text pipeline); revenue potential and fit are high. Slower time-to-revenue (need trained projector + positioning).
- **4. Prediction markets & trading-adjacent** — 3.2. Very high revenue potential but slow time-to-revenue, regulatory/sales friction, and higher ops (data, compliance). Fit is partial (Figgie/Polymarket semantics).
- **5. Library / platform licensing** — 3.7. Fast to revenue and low ops once the library is productized; high fit (reorg/Hex). Revenue ceiling and differentiation are moderate.
- **6. Consulting & custom pipelines** — 3.4. Good fit and revenue per engagement; high ops and variable sales friction. Differentiation is execution, not product.
- **7. Low-latency / edge positioning** — 3.2. Niche (ad-tech, real-time); moderate scores across the board, so judge is lower despite technical strength.

---

## Rank by judge (best first)


| Rank | Track                                     | Judge |
| ---- | ----------------------------------------- | ----- |
| 1    | Recommendation API / SaaS                 | 4.2   |
| 2    | Gaming & digital storefronts              | 3.8   |
| 2    | Multimodal catalog / “recommend by image” | 3.8   |
| 4    | Library / platform licensing              | 3.7   |
| 5    | Consulting & custom pipelines             | 3.4   |
| 6    | Prediction markets & trading-adjacent     | 3.2   |
| 6    | Low-latency / edge positioning            | 3.2   |


---

## Suggested use

- **Primary:** Recommendation API / SaaS (rank 1).
- **Secondary (pick one):** Gaming & storefronts **or** Multimodal catalog, depending on whether you prioritize distribution (gaming) or differentiation (vision).
- **Later:** Library licensing and consulting as leverage and services; prediction markets and low-latency edge once core product is live.

