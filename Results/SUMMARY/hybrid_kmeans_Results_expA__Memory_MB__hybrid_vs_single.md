| DatasetName                        |   NumClusters |   Single_Memory_MB |   Hybrid_Memory_MB |   Rel_Memory_MB |   Improvement_% |
|:-----------------------------------|--------------:|-------------------:|-------------------:|----------------:|----------------:|
| SYNTH_C_30_F_50_n1000_000k_kmeans  |           100 |                200 |                600 |               3 |            -200 |
| SYNTH_C_5_F_50_n1000_000k_kmeans   |           100 |                200 |                600 |               3 |            -200 |
| SYNTH_C_80_F_120_n1000_000k_kmeans |           100 |                480 |               1440 |               3 |            -200 |

**Summary**

| baseline   | compare_suite   | metric    |   n_pairs |   mean_baseline |   mean_compare |   mean_rel |   mean_improvement_% |   t_test_stat |   t_test_p |   wilcoxon_stat |   wilcoxon_p |   cohens_d |
|:-----------|:----------------|:----------|----------:|----------------:|---------------:|-----------:|---------------------:|--------------:|-----------:|----------------:|-------------:|-----------:|
| Single     | Hybrid          | Memory_MB |         3 |         293.333 |            880 |          3 |                 -200 |      -3.14286 |  0.0880707 |               0 |         0.25 |   -1.81453 |