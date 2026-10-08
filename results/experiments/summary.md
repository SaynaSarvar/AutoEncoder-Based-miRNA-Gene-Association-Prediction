# Step 8 experiment summary

Jobs: done 421

## Replication: paper Table 2 (lstm, random holdout, by latent size)

| latent | latent_only | paper | original_plus_latent | paper |
|---|---|---|---|---|
| 8 | 0.7990 | 0.9022 | 0.9499 | 0.9818 |
| 16 | 0.8003 | 0.8988 | 0.9512 | 0.9837 |
| 32 | 0.8612 | 0.9229 | 0.9515 | 0.9832 |
| 64 | 0.9351 | 0.9231 | 0.9510 | 0.9846 |
| 128 | 0.9355 | 0.9331 | 0.9521 | 0.9840 |

ROC-AUC. 'n/a' = paper value not entered in configs (paper_reference).

## Replication: paper Tables 3 and 11 (lstm, 5-fold CV, latent 64)

original_plus_latent = AEmiGAP (paper Table 3 average ± std over its 5 folds); original_only = miGAP (Table 11, average only). Paper folds are numbered 1–5, ours 0–4: compare mean ± std, not fold by fold. The paper reports no PR-AUC.

| input | metric | ours (mean ± std, 5 folds) | paper |
|---|---|---|---|
| original_plus_latent | roc_auc | 0.9520 ± 0.0006 (n=5) | 0.9846 ± 0.0018 |
| original_plus_latent | pr_auc | 0.9588 ± 0.0005 (n=5) | n/a |
| original_plus_latent | accuracy | 0.8776 ± 0.0007 (n=5) | 0.9456 ± 0.0033 |
| original_plus_latent | f1 | 0.8742 ± 0.0011 (n=5) | 0.9466 ± 0.0034 |
| original_plus_latent | precision | 0.8990 ± 0.0049 (n=5) | 0.9658 ± 0.0147 |
| original_plus_latent | recall | 0.8508 ± 0.0061 (n=5) | 0.9284 ± 0.0080 |
| original_only | roc_auc | 0.9499 ± 0.0002 (n=5) | 0.9804 |
| original_only | pr_auc | 0.9570 ± 0.0002 (n=5) | n/a |
| original_only | accuracy | 0.8749 ± 0.0001 (n=5) | 0.9340 |
| original_only | f1 | 0.8716 ± 0.0009 (n=5) | 0.9326 |
| original_only | precision | 0.8953 ± 0.0056 (n=5) | 0.9380 |
| original_only | recall | 0.8491 ± 0.0069 (n=5) | 0.9278 |

## Replication: paper Table 13 (models, random holdout, original_plus_latent, z=64)

| model | ROC-AUC | PR-AUC | acc | F1 | best/run epochs | paper ROC-AUC | note |
|---|---|---|---|---|---|---|---|
| lstm | 0.9510 | 0.9581 | 0.8767 | 0.8739 | 6/11 | 0.9846 |  |
| bilstm | 0.9551 | 0.9616 | 0.8821 | 0.8799 | 8/13 | 0.9638 |  |
| gru | 0.9479 | 0.9555 | 0.8726 | 0.8684 | 6/11 | 0.9478 |  |
| transformer | 0.9582 | 0.9654 | 0.8885 | 0.8838 | 42/47 | 0.9436 |  |
| lstm_timesteps | 0.5342 | 0.5153 | 0.5000 | 0.6667 | 1/3 | n/a | capped: batch 1024, max 3 epochs |
| mlp | 0.9507 | 0.9592 | 0.8768 | 0.8688 | 49/50 | n/a | max 50 epochs |

## Robustness: train strategy x test strategy (ROC-AUC)

Rows = training negatives, columns = test negatives. MLP = mean ± std over 5 folds; floor = best of shortcut baselines a–c (mean over folds, same cell); LSTM = fold 0 only.

### random_kfold

| train \ test | paper | random | degree_aware |
|---|---|---|---|
| **paper** | MLP 0.9511 ± 0.0006<br>floor 0.893 (degree)<br>LSTM f0 0.9510 | MLP 0.6554 ± 0.0021<br>floor 0.879 (degree)<br>LSTM f0 0.6681 | MLP 0.5159 ± 0.0007<br>floor 0.543 (degree)<br>LSTM f0 0.5188 |
| **random** | MLP 0.8659 ± 0.0021<br>floor 0.893 (degree)<br>LSTM f0 0.8857 | MLP 0.8334 ± 0.0027<br>floor 0.880 (degree)<br>LSTM f0 0.8713 | MLP 0.5413 ± 0.0020<br>floor 0.543 (degree)<br>LSTM f0 0.5690 |
| **degree_aware** | MLP 0.6198 ± 0.0116<br>floor 0.825 (degree)<br>LSTM f0 0.6425 | MLP 0.5982 ± 0.0049<br>floor 0.801 (degree)<br>LSTM f0 0.6370 | MLP 0.5875 ± 0.0026<br>floor 0.551 (degree)<br>LSTM f0 0.6217 |

### cold_mirna

| train \ test | paper | random | degree_aware |
|---|---|---|---|
| **paper** | MLP 0.8887 ± 0.0083<br>floor 0.858 (distances)<br>LSTM f0 0.8951 | MLP 0.5932 ± 0.0041<br>floor 0.786 (degree)<br>LSTM f0 0.6030 | MLP 0.5218 ± 0.0036<br>floor 0.551 (degree)<br>LSTM f0 0.5210 |
| **random** | MLP 0.6969 ± 0.0193<br>floor 0.812 (degree)<br>LSTM f0 0.7265 | MLP 0.6432 ± 0.0200<br>floor 0.786 (degree)<br>LSTM f0 0.6999 | MLP 0.5304 ± 0.0042<br>floor 0.551 (degree)<br>LSTM f0 0.5369 |
| **degree_aware** | MLP 0.5961 ± 0.0182<br>floor 0.812 (degree)<br>LSTM f0 0.6093 | MLP 0.5736 ± 0.0119<br>floor 0.786 (degree)<br>LSTM f0 0.6032 | MLP 0.5664 ± 0.0063<br>floor 0.551 (degree)<br>LSTM f0 0.5822 |

### cold_gene

| train \ test | paper | random | degree_aware |
|---|---|---|---|
| **paper** | MLP 0.9256 ± 0.0032<br>floor 0.858 (distances)<br>LSTM f0 0.9216 | MLP 0.6350 ± 0.0067<br>floor 0.794 (degree)<br>LSTM f0 0.6396 | MLP 0.5165 ± 0.0020<br>floor 0.513 (log_length)<br>LSTM f0 0.5190 |
| **random** | MLP 0.8329 ± 0.0029<br>floor 0.799 (distances)<br>LSTM f0 0.7981 | MLP 0.7930 ± 0.0028<br>floor 0.794 (degree)<br>LSTM f0 0.7728 | MLP 0.5203 ± 0.0020<br>floor 0.514 (distances)<br>LSTM f0 0.5164 |
| **degree_aware** | MLP 0.5905 ± 0.0087<br>floor 0.797 (degree)<br>LSTM f0 0.5841 | MLP 0.5668 ± 0.0040<br>floor 0.794 (degree)<br>LSTM f0 0.5684 | MLP 0.5628 ± 0.0012<br>floor 0.514 (distances)<br>LSTM f0 0.5675 |

## Strong-evidence AUC and per-miRNA AUC (own test negatives)

| split | train=test strategy | model | ROC-AUC | strong-evidence AUC | per-miRNA AUC (n miRNAs) |
|---|---|---|---|---|---|
| random_kfold | paper | mlp | 0.9511 ± 0.0006 | 0.8913 ± 0.0034 | 0.9126 ± 0.0013 (1996) |
| random_kfold | paper | lstm f0 | 0.9510 | 0.8937 | 0.9158 (2002) |
| random_kfold | paper | logreg original_only | 0.6845 ± 0.0001 | 0.6359 ± 0.0065 | 0.6435 ± 0.0009 (1996) |
| random_kfold | paper | floor: distances | 0.8585 ± 0.0004 | 0.8318 ± 0.0022 | 0.8531 ± 0.0012 (1996) |
| random_kfold | paper | floor: degree | 0.8934 ± 0.0003 | 0.7563 ± 0.0047 | 0.8068 ± 0.0003 (1996) |
| random_kfold | random | mlp | 0.8334 ± 0.0027 | 0.6798 ± 0.0068 | 0.6833 ± 0.0072 (1996) |
| random_kfold | random | lstm f0 | 0.8713 | 0.7342 | 0.7777 (2002) |
| random_kfold | random | logreg original_only | 0.6123 ± 0.0011 | 0.5113 ± 0.0088 | 0.5554 ± 0.0011 (1996) |
| random_kfold | random | floor: distances | 0.5604 ± 0.0010 | 0.5357 ± 0.0070 | 0.5444 ± 0.0012 (1996) |
| random_kfold | random | floor: degree | 0.8799 ± 0.0006 | 0.7331 ± 0.0050 | 0.7782 ± 0.0013 (1996) |
| random_kfold | degree_aware | mlp | 0.5875 ± 0.0026 | 0.5655 ± 0.0123 | 0.5706 ± 0.0016 (1914) |
| random_kfold | degree_aware | lstm f0 | 0.6217 | 0.5755 | 0.5929 (1919) |
| random_kfold | degree_aware | logreg original_only | 0.5081 ± 0.0012 | 0.5078 ± 0.0069 | 0.5055 ± 0.0006 (1914) |
| random_kfold | degree_aware | floor: distances | 0.5144 ± 0.0011 | 0.4935 ± 0.0064 | 0.5011 ± 0.0016 (1914) |
| random_kfold | degree_aware | floor: degree | 0.5509 ± 0.0006 | 0.4736 ± 0.0036 | 0.5197 ± 0.0014 (1914) |
| cold_mirna | paper | mlp | 0.8887 ± 0.0083 | 0.8578 ± 0.0189 | 0.8980 ± 0.0093 (460) |
| cold_mirna | paper | lstm f0 | 0.8951 | 0.8727 | 0.9108 (461) |
| cold_mirna | paper | logreg original_only | 0.6675 ± 0.0062 | 0.6145 ± 0.0283 | 0.6394 ± 0.0043 (460) |
| cold_mirna | paper | floor: distances | 0.8585 ± 0.0028 | 0.8314 ± 0.0063 | 0.8508 ± 0.0042 (460) |
| cold_mirna | paper | floor: degree | 0.8118 ± 0.0017 | 0.7429 ± 0.0076 | 0.8027 ± 0.0014 (460) |
| cold_mirna | random | mlp | 0.6432 ± 0.0200 | 0.6066 ± 0.0300 | 0.6635 ± 0.0109 (460) |
| cold_mirna | random | lstm f0 | 0.6999 | 0.6963 | 0.7551 (461) |
| cold_mirna | random | logreg original_only | 0.5878 ± 0.0043 | 0.4895 ± 0.0340 | 0.5543 ± 0.0020 (460) |
| cold_mirna | random | floor: distances | 0.5602 ± 0.0032 | 0.5352 ± 0.0164 | 0.5418 ± 0.0038 (460) |
| cold_mirna | random | floor: degree | 0.7863 ± 0.0012 | 0.7174 ± 0.0070 | 0.7742 ± 0.0012 (460) |
| cold_mirna | degree_aware | mlp | 0.5664 ± 0.0063 | 0.5644 ± 0.0084 | 0.5563 ± 0.0059 (460) |
| cold_mirna | degree_aware | lstm f0 | 0.5822 | 0.5468 | 0.5611 (461) |
| cold_mirna | degree_aware | logreg original_only | 0.5081 ± 0.0031 | 0.5071 ± 0.0097 | 0.5026 ± 0.0017 (460) |
| cold_mirna | degree_aware | floor: distances | 0.5143 ± 0.0039 | 0.4927 ± 0.0201 | 0.4979 ± 0.0032 (460) |
| cold_mirna | degree_aware | floor: degree | 0.5507 ± 0.0012 | 0.4816 ± 0.0069 | 0.5113 ± 0.0036 (460) |
| cold_gene | paper | mlp | 0.9256 ± 0.0032 | 0.8572 ± 0.0164 | 0.8688 ± 0.0076 (1987) |
| cold_gene | paper | lstm f0 | 0.9216 | 0.8647 | 0.8662 (1963) |
| cold_gene | paper | logreg original_only | 0.6808 ± 0.0047 | 0.6321 ± 0.0139 | 0.6392 ± 0.0073 (1987) |
| cold_gene | paper | floor: distances | 0.8585 ± 0.0049 | 0.8312 ± 0.0188 | 0.8534 ± 0.0069 (1987) |
| cold_gene | paper | floor: degree | 0.7972 ± 0.0011 | 0.6252 ± 0.0056 | 0.5000 ± 0.0000 (1987) |
| cold_gene | random | mlp | 0.7930 ± 0.0028 | 0.6292 ± 0.0123 | 0.5664 ± 0.0082 (1987) |
| cold_gene | random | lstm f0 | 0.7728 | 0.6232 | 0.5678 (1963) |
| cold_gene | random | logreg original_only | 0.6091 ± 0.0019 | 0.5090 ± 0.0143 | 0.5490 ± 0.0025 (1987) |
| cold_gene | random | floor: distances | 0.5604 ± 0.0106 | 0.5353 ± 0.0369 | 0.5445 ± 0.0114 (1987) |
| cold_gene | random | floor: degree | 0.7939 ± 0.0018 | 0.6206 ± 0.0064 | 0.5000 ± 0.0000 (1987) |
| cold_gene | degree_aware | mlp | 0.5628 ± 0.0012 | 0.5455 ± 0.0197 | 0.5493 ± 0.0020 (1909) |
| cold_gene | degree_aware | lstm f0 | 0.5675 | 0.5640 | 0.5535 (1905) |
| cold_gene | degree_aware | logreg original_only | 0.5047 ± 0.0020 | 0.5014 ± 0.0152 | 0.5044 ± 0.0013 (1909) |
| cold_gene | degree_aware | floor: distances | 0.5143 ± 0.0015 | 0.4930 ± 0.0308 | 0.5011 ± 0.0016 (1909) |
| cold_gene | degree_aware | floor: degree | 0.4995 ± 0.0021 | 0.2716 ± 0.0080 | 0.5000 ± 0.0000 (1909) |

per-miRNA AUC: mean ROC-AUC within each test miRNA with >=10 positives and >=10 negatives (cold_mirna splits have no such miRNA in common with training, which is fine: the metric is computed on test rows only).
