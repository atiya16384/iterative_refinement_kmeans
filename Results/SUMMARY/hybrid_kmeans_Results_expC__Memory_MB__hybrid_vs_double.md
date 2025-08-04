| DatasetName                        |   NumClusters |   Double_Memory_MB |   Hybrid_Memory_MB |   Rel_Memory_MB |   Improvement_% |
|:-----------------------------------|--------------:|-------------------:|-------------------:|----------------:|----------------:|
| SYNTH_C_30_F_50_n1000_000k_kmeans  |           100 |                4   |                6   |             1.5 |             -50 |
| SYNTH_C_5_F_50_n1000_000k_kmeans   |           100 |                4   |                6   |             1.5 |             -50 |
| SYNTH_C_80_F_120_n1000_000k_kmeans |           100 |                9.6 |               14.4 |             1.5 |             -50 |

**Summary**

| baseline   | compare_suite   | metric    |   n_pairs |   mean_baseline |   mean_compare |   mean_rel |   mean_improvement_% |   t_test_stat |   t_test_p |   wilcoxon_stat |   wilcoxon_p |   cohens_d |
|:-----------|:----------------|:----------|----------:|----------------:|---------------:|-----------:|---------------------:|--------------:|-----------:|----------------:|-------------:|-----------:|
| Double     | Hybrid          | Memory_MB |         3 |         5.86667 |            8.8 |        1.5 |                  -50 |      -3.14286 |  0.0880707 |               0 |         0.25 |   -1.81453 |