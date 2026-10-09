# Results (all numbers from results/results056.json; sd = sample sd over 3 seeds, ddof=1; RF/entity rows are single runs)
Lock commit 6b5b31a (PREREG.md + src/run056.py) precedes all scoring in the commit chain (dates author-set). Gates applied exactly as preregistered.

## Run history and deviations (disclosed)
- First launch died silently after D1 (no scores); relaunched unchanged. Second launch was OOM-killed (exit 137) after the two official-split blocks were scored; a memory-reducing change (float32 mean/sd and normalisation, gc) was committed; this changes numerics slightly (near-identical, not bit-identical) for the cold-split, D8/D8b and D9 stages, and cannot move any gate (closest margin: BioSNAP D5, drop 0.116 vs the 0.15 gate), then the run resumed from the checkpoint. Official-split numbers come from the pre-fix code path; all later stages from the fixed one. Same seeds/hyperparameters.
- D8 gene-side coverage was 0%: BioSNAP "Gene" holds UniProt accessions, Hetionet uses gene symbols (my mapping bug, seen only after D8 scored). D8b addendum (src/run056_d8b.py, committed before running) maps accession to primary gene symbol via the UniProt REST API (2,175 of 2,182 mapped) and reruns the same code and gate. D8 (original) is kept and marked as invalid-coverage; D8b is the one interpreted, with the mapping source being a post-hoc addition.

## D1 data audit
4 unique unparsable SMILES. Test pairs whose drug AND protein both appear in train: BindingDB 79.2%, BioSNAP 97.0% (drug seen 79.8% / 97.3%; protein seen 98.0% / 99.8%). BindingDB train is 50% positive, val/test 14.0% / 14.3%.

## D2 headline (M1 = ECFP + ESM-2-8M + MLP, official test split) - gate >= 0.85 on both: MET
BindingDB AUROC 0.898 (sd 0.0015), BioSNAP 0.897 (sd 0.0036).
## D3 baselines - gate "M1 beats best baseline by >= 0.01 on both": NOT MET
BindingDB: M1 0.898, k-mer MLP 0.909 (sd 0.0012), RF on [ECFP,ESM] 0.923. BioSNAP: M1 0.897, k-mer MLP 0.886, RF 0.891 (margin +0.007). A simple RF beats the MLP on BindingDB.
## D4 shortcut controls - gate "shortcut material" (>= 0.75): FIRES
Drug-only RF / protein-only RF AUROC: BindingDB 0.824 / 0.833; BioSNAP 0.645 / 0.809.
## D5 cold start (pooled pairs, seed 0, M1 AUROC; sd over seeds) - gate "fragile" (cold-both < official - 0.15 on either): FIRES (BindingDB)
BindingDB: cold-drug 0.839 (0.0028), cold-protein 0.801 (0.0084), cold-both 0.677 (0.0116) vs 0.898 official (drop 0.221). BioSNAP: 0.851 (0.0063), 0.859 (0.0043), cold-both 0.781 (0.0024) (drop 0.116, gate not met there). Cold-both test sets are small (1,398 / 1,183 pairs) and one random drug/protein holdout draw.
## D6 does the protein LM help (ESM vs k-mer, same MLP) - gate gain >= 0.02 in cold-protein AND cold-both: MET on BioSNAP only
BindingDB: ESM minus k-mer = -0.039 (cold-protein), -0.056 (cold-both); k-mer is better. BioSNAP: +0.037 and +0.048.
## D7 distribution shift (BindingDB test, 14.3% positive)
M1 3-seed ENSEMBLE AUPRC 0.605 (the 3-seed mean of single-model AUPRC reported in D2/results JSON is 0.583) vs 0.143 constant score; accuracy 0.820 at 0.5, 0.892 at the val-picked threshold 0.95 (accuracy alone is a weak metric at this prevalence). BioSNAP: AUPRC 0.908 vs 0.504.
## D8/D8b knowledge-graph features (D8b is a POST-HOC addendum written after D8 failed on coverage; it flipped the gate from not met to met. Do not read it as 'KG adds': 35% drug coverage, approved-drug/missingness confound, and overlap with the entity-memorisation shortcut. Original D8 row kept.) (BioSNAP, no binds-type edges) - gate gain >= 0.02 on cold-both
D8 original (gene coverage 0%, drug coverage 34.9%): cold-both 0.781 -> 0.791 (+0.010): not met, but invalid for the gene side. D8b (gene coverage 97.5%, drug 34.9%): cold-both 0.781 -> 0.806 (+0.025; sd 0.0024 / 0.0040): gate met; official 0.897 -> 0.915. CAVEAT: only 35% of rows have a drug KG vector and unmapped rows get zeros; no control for coverage, node degree or "has-a-KG-entry" as an informative indicator was run, so the gain is not attributed to KG content.
## D9 NR transfer (Yamanishi; all 54 drugs and 26 proteins rebuilt from KEGG, 1,404 pairs, 90 positives = 6.4%) - gate AUROC >= 0.70 and above protein-only: NOT MET
M1 trained on BindingDB train+val: AUROC 0.517, AUPRC 0.081 (base rate 0.064). Protein-only RF: AUROC 0.637, AUPRC 0.106. Zero transfer signal for M1.
## D10 null and leakage
(a) permuted-label null AUROC 0.488 (sd 0.014) BindingDB, 0.499 (0.014) BioSNAP: sane (within 0.50 +/- 0.03). (b) "entity memorisation" score (mean train label of the pair's drug and protein, no chemistry/sequence): AUROC 0.881 BindingDB (M1 0.898), 0.784 BioSNAP.

## Limits
Not the authors' model: no drug language model, no heterogeneous GNN, small ESM-2 8M, Hetionet instead of their graph. "Finds unknown interactions" not tested. Few seeds, one cold-split draw, no hyperparameter search, results are for this pipeline only. Underpowered where noted (cold-both sets, NR 90 positives).


## Provenance
All commits are author-dated. The repo was first pushed to GitHub after the results commit, so a public server time of the lock cannot be shown. Lock-before-score rests on commit order in one pushed history plus my own run logs (results/run056*.log); this is not a verified preregistration.
