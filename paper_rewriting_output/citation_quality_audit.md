# Citation Quality Audit

- Output directory: `paper_rewriting_output`
- Scene: journal
- Target citation count: 25
- Entries analyzed: 30
- Verified: 30 | Mismatched: 0 | Dead: 0
- Overall quality score: 96/100
- Status: PASS

> Each entry below includes a teaching note explaining *why* the citation quality matters.

## Per-Citation Analysis

| ID | DOI | Type | Resolves | Title Match | Year Match | Score | Status |
|---|---|---|---|---|---|---|---|
| C001 | 10.1016/j.swevo.2026.102288 | survey | yes | 100% | yes | 100 | verified |
| C002 | 10.1016/j.swevo.2026.102288 | survey | yes | 100% | yes | 100 | verified |
| C003 | 10.1016/j.swevo.2026.102288 | survey | yes | 100% | yes | 100 | verified |
| C004 | 10.1016/j.swevo.2026.102288 | survey | yes | 100% | yes | 100 | verified |
| C005 | 10.1016/j.swevo.2026.102288 | survey | yes | 100% | yes | 100 | verified |
| C006 | 10.1016/j.swevo.2026.102288 | survey | yes | 100% | yes | 100 | verified |
| C007 | 10.1016/j.swevo.2025.101894 | sota | yes | 100% | yes | 95 | verified |
| C008 | 10.1016/j.swevo.2025.101894 | sota | yes | 100% | yes | 95 | verified |
| C009 | 10.1016/j.swevo.2025.101894 | sota | yes | 100% | yes | 95 | verified |
| C010 | 10.1016/j.swevo.2025.101894 | sota | yes | 100% | yes | 95 | verified |
| C011 | 10.1016/j.swevo.2025.101894 | sota | yes | 100% | yes | 95 | verified |
| C012 | 10.1109/TEVC.2023.3346672 | sota | yes | 100% | yes | 95 | verified |
| C013 | 10.1109/TEVC.2023.3346672 | sota | yes | 100% | yes | 95 | verified |
| C014 | 10.1109/TEVC.2023.3346672 | sota | yes | 100% | yes | 95 | verified |
| C015 | 10.1109/TEVC.2023.3346672 | sota | yes | 100% | yes | 95 | verified |
| C016 | 10.1109/TEVC.2023.3346672 | sota | yes | 100% | yes | 95 | verified |
| C017 | 10.1016/j.swevo.2025.102071 | sota | yes | 100% | yes | 95 | verified |
| C018 | 10.1016/j.swevo.2025.102071 | sota | yes | 100% | yes | 95 | verified |
| C019 | 10.1016/j.swevo.2025.102071 | sota | yes | 100% | yes | 95 | verified |
| C020 | 10.1016/j.swevo.2025.102071 | sota | yes | 100% | yes | 95 | verified |
| C021 | 10.1016/j.swevo.2024.101838 | survey | yes | 100% | yes | 95 | verified |
| C022 | 10.1016/j.swevo.2024.101838 | survey | yes | 100% | yes | 95 | verified |
| C023 | 10.1016/j.swevo.2024.101838 | survey | yes | 100% | yes | 95 | verified |
| C024 | 10.1016/j.swevo.2024.101838 | survey | yes | 100% | yes | 95 | verified |
| C025 | 10.1016/j.ins.2024.121134 | sota | yes | 100% | yes | 95 | verified |
| C026 | 10.1016/j.ins.2024.121134 | sota | yes | 100% | yes | 95 | verified |
| C027 | 10.1016/j.ins.2024.121134 | sota | yes | 100% | yes | 95 | verified |
| C028 | 10.1016/j.ins.2024.121134 | sota | yes | 100% | yes | 95 | verified |
| C029 | 10.1016/j.asoc.2024.111952 | sota | yes | 100% | yes | 95 | verified |
| C030 | 10.1016/j.asoc.2024.111952 | sota | yes | 100% | yes | 95 | verified |

## Citation Diversity Gaps

**Missing foundational method or theory papers.** Only 0 of 30 entries (0%). Cite the 2-3 methods your work builds on. Explain inheritance clearly. Consider adding 1-3 foundational method or theory paper references.

**Missing dataset, benchmark, or evaluation protocol papers.** Only 0 of 30 entries (0%). Cite the datasets you evaluate on. Report dataset statistics. Consider adding 1-3 dataset, benchmark, or evaluation protocol paper references.

**Missing domain-application or impact papers.** Only 0 of 30 entries (0%). Optional unless your contribution is application-motivated. Consider adding 1-3 domain-application or impact paper references.


## Scene-Specific Citation Strategy

For **journal** papers, your citation strategy should:

- **direct task or state-of-the-art paper**: Must cite the 3-5 most recent competing methods. Missing these is a desk-reject risk.
- **foundational method or theory paper**: Cite the 2-3 methods your work builds on. Explain inheritance clearly.
- **dataset, benchmark, or evaluation protocol paper**: Cite the datasets you evaluate on. Report dataset statistics.
- **survey, review, or meta-analysis**: Cite 1-2 recent surveys to position your work in the broader landscape.
- **domain-application or impact paper**: Optional unless your contribution is application-motivated.
- **limitation, robustness, reproducibility, or ethics paper**: Include 1-2 limitation/robustness papers to show awareness of field challenges.

## Citation Strategy Principles

- **Diversity over density.** A narrow citation pool makes your Introduction read as insular. Mix SOTA, foundational, benchmark, survey, and application papers.
- **Recency signals engagement.** Most citations should be from the last 3 years. Older citations are fine for foundational work, but they need a reason to be there.
- **Verifiability is non-negotiable.** Every DOI must resolve. A dead DOI in your final paper is a credibility failure that reviewers notice immediately.
- **Type matters by venue.** Journals expect deep SOTA coverage. Reports expect broad survey coverage. Competitions expect benchmark and leaderboard coverage. Match your strategy to your scene.
