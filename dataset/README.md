# Dataset

Redistributed from Zenodo under CC BY 4.0.

**Source:** Wakai, A. (2025). *Historical precipitation and flood damage in
Japan: functional data analysis and evaluation of models.* Zenodo.
<https://doi.org/10.5281/zenodo.14015790>

## Files used by this project

| File | Role |
|---|---|
| `data/DB_input+res_ptn02d14_logDmgPop.csv` | the analysis input — 5704 events, 40 columns |
| `data_structures.xlsx` | variable dictionary |
| `readme.docx` | original dataset documentation |

Other CSVs in `data/` are outputs of the source study's own functional data
analysis and are not used here.

## Coverage

Kanto region, Japan — Ibaraki (08), Chiba (12) and Kanagawa (14) prefectures,
plus seven nationally managed Class-A river systems. 75 water systems, 962
rivers, 1993–2020.

## Key columns

| Column | Meaning |
|---|---|
| `wsysCd`, `rivCd` | water system and river identifiers |
| `year`, `date` | event timing |
| `area` | river basin area (km²) |
| `slope` | average basin slope ratio |
| `population` | population within the basin |
| `29d` … `0d` | mean daily precipitation on each of the 30 days before the event |
| `damageObs` | observed flood damage (raw) — dropped, it is the target's numerator |
| `log₁₀(dmgObs/pop)` | **target variable** |
| `log₁₀(dmgPred/pop)` | source study's own prediction — dropped to prevent leakage |

## Note

The source study found 14 days to be its optimal antecedent precipitation window
and split the record into two periods (1993–1999 and 2000–2020). This project
uses a different window and a single chronological split; see the root README.
