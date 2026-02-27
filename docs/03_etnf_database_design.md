# ETNF database design

Essential Tuple Normal Form (ETNF) and how it guides the Clickstream schema. For the concrete schema and Dublin Core usage in this repo, see [04 FOSS datasets and Dublin Core](04_foss_datasets_etnf_dublin_core_xmp.md).

---

## What is ETNF?

**Essential Tuple Normal Form** (Darwen, Date, Fagin, 2012) sits strictly between 4NF and 5NF. A relation is in ETNF iff it is in **BCNF** and **every explicitly declared join dependency has a component that is a superkey**. ETNF removes tuple redundancy as effectively as 5NF.

---

## Design steps

1. **BCNF** — Start from Boyce-Codd Normal Form: every determinant is a superkey; no partial or transitive dependency anomalies.

2. **Join dependencies** — Identify attribute groups that are always used together (e.g. from SELECT/UPDATE/INSERT patterns).

3. **ETNF** — Split relations by access and update patterns so that every declared join dependency has a superkey component; no redundant tuples.

**Reference:** Darwen, H.; Date, C.J.; Fagin, R. (2012). “A normal form for preventing redundant tuples in relational databases.” _ICDT 2012._ doi:10.1145/2274576.2274589

---

## See also

- [04 FOSS datasets and Dublin Core](04_foss_datasets_etnf_dublin_core_xmp.md) — Schema and Dublin Core/XMP in this repo.
- [00 RecGPT library](00_recgpt_library.md) — Clickstream.Fetch and Repo.
