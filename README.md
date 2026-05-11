# QTSF-Quantum-Classical-Framework-for-Time-Series-Forecasting
Multivariate Time Series Forecasting using Quantum Classical Method



(venv) sanch@balawar:~/Documents/Time-Series-Library-main$ python vqff_etth2_fast.py     --data_path ~/Documents/Time-Series-Library-main/dataset/ETT-small/ETTh2.csv     --device cuda:0     --pred_lens 96 192 336 720     --batch_size 32 --epochs 10 --patience 3



(venv) sanch@balawar:~/Documents/Time-Series-Library-main$ python vqff_ettm1_fast.py     --data_path ~/Documents/Time-Series-Library-main/dataset/ETT-small/ETTm1.csv  --device cuda:0     --pred_lens 96 192 336 720     --batch_size 32 --epochs 10 --patience 3




(venv) sanch@balawar:~/Documents/Time-Series-Library-main$ python vqff_exchange_fast.py \
    --data_path ~/Documents/Time-Series-Library-main/dataset/exchange_rate/exchange_rate.csv \
    --device cuda:0 \
    --pred_lens 96 192 336 720 \
    --batch_size 32 --epochs 10 --patience 3


  [GPU] NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition  |  VRAM 102.0 GB  |  CUDA 13.0
========================================================================
  VQF-F (broadcast): Variational Quantum Forecaster on Exchange Rate
  KEY FIX: PennyLane native parameter broadcasting
           Recommended backend: default.qubit (omit --use_lightning)
========================================================================
  data_path  : /home/sanch/Documents/Time-Series-Library-main/dataset/exchange_rate/exchange_rate.csv
  pred_lens  : [96, 192, 336, 720]
  batch_size : 32  seq_len: 96
  epochs     : 10  patience: 3
  N_QUBITS   : 6  depth: 2
  device     : cuda:0  AMP: True
  lightning  : False
  freq       : daily  (3 time features: month/day/weekday)
  Split      : 70/10/20 %  (TSLib Dataset_Custom ratio)

  Exchange Rate shape: (7588, 9)  columns: ['date', '0', '1', '2', '3', '4', '5', '6', 'OT']

========================================================================
  VQF-F (broadcast) | Exchange Rate | pred_len=96 | enc_in=8 | device=cuda:0
  n_qubits=6 | depth=2 | batch=32 | AMP=True
========================================================================
    [train]  rows: 0..5311  (5311 total)  channels: 8
    [val]  rows: 5215..6071  (856 total)  channels: 8
    [test]  rows: 5975..7588  (1613 total)  channels: 8
  Windows → train:5120  val:665  test:1422
  [Quantum] default.qubit (6 qubits, broadcasted)  [FASTEST for small circuits]
  Params → Q(CPU):192  C(GPU):68368
  Quantum calls per batch: 8  (was 256 in original)
  GPU mem  alloc=0MB  reserved=2MB

    Ep |   Train MSE |     Val MSE |         LR |  Time(s)
  --------------------------------------------------------
    [Epoch 1] training 160 batches...
      batch    1/160  loss=0.33835  elapsed=2.0s  ETA=313.3s
      batch   16/160  loss=0.17759  elapsed=25.3s  ETA=227.6s
      batch   32/160  loss=0.26275  elapsed=50.4s  ETA=201.6s
      batch   48/160  loss=0.21056  elapsed=75.6s  ETA=176.5s
      batch   64/160  loss=0.23671  elapsed=101.0s  ETA=151.4s
      batch   80/160  loss=0.22259  elapsed=126.2s  ETA=126.2s
      batch   96/160  loss=0.26097  elapsed=151.3s  ETA=100.9s
      batch  112/160  loss=0.20631  elapsed=176.3s  ETA=75.5s
      batch  128/160  loss=0.25762  elapsed=201.3s  ETA=50.3s
      batch  144/160  loss=0.17737  elapsed=226.3s  ETA=25.1s
      batch  160/160  loss=0.27246  elapsed=251.4s  ETA=0.0s
     1 |     0.20630 |     0.18384 |   1.00e-04 |    252.4  GPU mem  alloc=19MB  reserved=25MB
    [Epoch 2] training 160 batches...
      batch    1/160  loss=0.15472  elapsed=1.7s  ETA=263.0s
      batch   16/160  loss=0.22872  elapsed=25.1s  ETA=225.6s
      batch   32/160  loss=0.15776  elapsed=50.2s  ETA=200.9s
      batch   48/160  loss=0.22239  elapsed=75.4s  ETA=175.9s
      batch   64/160  loss=0.15810  elapsed=100.6s  ETA=150.9s
      batch   80/160  loss=0.17938  elapsed=125.8s  ETA=125.8s
      batch   96/160  loss=0.30144  elapsed=151.0s  ETA=100.6s
      batch  112/160  loss=0.18609  elapsed=176.2s  ETA=75.5s
      batch  128/160  loss=0.19102  elapsed=201.4s  ETA=50.3s
      batch  144/160  loss=0.25374  elapsed=226.6s  ETA=25.2s
      batch  160/160  loss=0.21408  elapsed=251.8s  ETA=0.0s
     2 |     0.19183 |     0.17546 |   5.00e-05 |    252.6  GPU mem  alloc=19MB  reserved=25MB
    [Epoch 3] training 160 batches...
      batch    1/160  loss=0.18411  elapsed=1.5s  ETA=243.2s
      batch   16/160  loss=0.17952  elapsed=25.2s  ETA=226.6s
      batch   32/160  loss=0.21180  elapsed=50.3s  ETA=201.4s
      batch   48/160  loss=0.13737  elapsed=75.5s  ETA=176.1s
      batch   64/160  loss=0.11866  elapsed=100.7s  ETA=151.0s
      batch   80/160  loss=0.16600  elapsed=125.9s  ETA=125.9s
      batch   96/160  loss=0.17784  elapsed=151.1s  ETA=100.7s
      batch  112/160  loss=0.20695  elapsed=176.4s  ETA=75.6s
      batch  128/160  loss=0.24015  elapsed=201.6s  ETA=50.4s
      batch  144/160  loss=0.14449  elapsed=226.9s  ETA=25.2s
      batch  160/160  loss=0.21271  elapsed=252.2s  ETA=0.0s
     3 |     0.18407 |     0.17170 |   2.50e-05 |    253.0  GPU mem  alloc=19MB  reserved=25MB
    [Epoch 4] training 160 batches...
      batch    1/160  loss=0.19874  elapsed=1.6s  ETA=261.8s
      batch   16/160  loss=0.18111  elapsed=25.3s  ETA=227.5s
      batch   32/160  loss=0.23885  elapsed=50.6s  ETA=202.5s
      batch   48/160  loss=0.25530  elapsed=76.0s  ETA=177.3s
      batch   64/160  loss=0.12837  elapsed=101.3s  ETA=151.9s
      batch   80/160  loss=0.19829  elapsed=126.6s  ETA=126.6s
      batch   96/160  loss=0.16503  elapsed=151.9s  ETA=101.2s
      batch  112/160  loss=0.20302  elapsed=177.1s  ETA=75.9s
      batch  128/160  loss=0.13098  elapsed=202.3s  ETA=50.6s
      batch  144/160  loss=0.15843  elapsed=227.4s  ETA=25.3s
      batch  160/160  loss=0.25957  elapsed=252.6s  ETA=0.0s
     4 |     0.18069 |     0.17034 |   1.25e-05 |    253.5  GPU mem  alloc=19MB  reserved=25MB
    [Epoch 5] training 160 batches...
      batch    1/160  loss=0.14871  elapsed=1.5s  ETA=243.9s
      batch   16/160  loss=0.20895  elapsed=25.2s  ETA=226.7s
      batch   32/160  loss=0.21824  elapsed=50.4s  ETA=201.5s
      batch   48/160  loss=0.17631  elapsed=75.6s  ETA=176.4s
      batch   64/160  loss=0.17726  elapsed=100.8s  ETA=151.3s
      batch   80/160  loss=0.15256  elapsed=125.9s  ETA=125.9s
      batch   96/160  loss=0.16902  elapsed=151.1s  ETA=100.7s
      batch  112/160  loss=0.14830  elapsed=176.3s  ETA=75.6s
      batch  128/160  loss=0.17618  elapsed=201.6s  ETA=50.4s
      batch  144/160  loss=0.11615  elapsed=227.0s  ETA=25.2s
      batch  160/160  loss=0.16906  elapsed=252.2s  ETA=0.0s
     5 |     0.17912 |     0.16860 |   6.25e-06 |    253.0  GPU mem  alloc=19MB  reserved=25MB
    [Epoch 6] training 160 batches...
      batch    1/160  loss=0.19637  elapsed=1.6s  ETA=260.7s
      batch   16/160  loss=0.22053  elapsed=25.2s  ETA=227.0s
      batch   32/160  loss=0.23409  elapsed=50.5s  ETA=201.9s
      batch   48/160  loss=0.22122  elapsed=75.7s  ETA=176.6s
      batch   64/160  loss=0.12377  elapsed=100.9s  ETA=151.3s
      batch   80/160  loss=0.19176  elapsed=126.1s  ETA=126.1s
      batch   96/160  loss=0.21182  elapsed=151.3s  ETA=100.9s
      batch  112/160  loss=0.14383  elapsed=176.5s  ETA=75.6s
      batch  128/160  loss=0.15864  elapsed=201.6s  ETA=50.4s
      batch  144/160  loss=0.20503  elapsed=226.9s  ETA=25.2s
      batch  160/160  loss=0.24258  elapsed=252.0s  ETA=0.0s
     6 |     0.17847 |     0.16923 |   3.13e-06 |    252.9  GPU mem  alloc=19MB  reserved=25MB
    [Epoch 7] training 160 batches...
      batch    1/160  loss=0.20859  elapsed=1.5s  ETA=243.1s
      batch   16/160  loss=0.16960  elapsed=25.2s  ETA=226.4s
      batch   32/160  loss=0.17705  elapsed=50.4s  ETA=201.5s
      batch   48/160  loss=0.21145  elapsed=75.6s  ETA=176.3s
      batch   64/160  loss=0.24642  elapsed=100.8s  ETA=151.1s
      batch   80/160  loss=0.20691  elapsed=126.0s  ETA=126.0s
      batch   96/160  loss=0.20945  elapsed=151.1s  ETA=100.7s
      batch  112/160  loss=0.14666  elapsed=176.3s  ETA=75.6s
      batch  128/160  loss=0.13880  elapsed=201.4s  ETA=50.4s
      batch  144/160  loss=0.16631  elapsed=226.6s  ETA=25.2s
      batch  160/160  loss=0.14379  elapsed=251.6s  ETA=0.0s
     7 |     0.17807 |     0.16969 |   1.56e-06 |    252.5  GPU mem  alloc=19MB  reserved=25MB
    [Epoch 8] training 160 batches...
      batch    1/160  loss=0.12037  elapsed=1.6s  ETA=261.1s
      batch   16/160  loss=0.10619  elapsed=25.1s  ETA=226.3s
      batch   32/160  loss=0.12366  elapsed=50.3s  ETA=201.3s
      batch   48/160  loss=0.15625  elapsed=75.5s  ETA=176.1s
      batch   64/160  loss=0.17340  elapsed=100.7s  ETA=151.0s
      batch   80/160  loss=0.16761  elapsed=125.8s  ETA=125.8s
      batch   96/160  loss=0.17369  elapsed=151.0s  ETA=100.7s
      batch  112/160  loss=0.21747  elapsed=176.2s  ETA=75.5s
      batch  128/160  loss=0.19843  elapsed=201.4s  ETA=50.3s
      batch  144/160  loss=0.17175  elapsed=226.6s  ETA=25.2s
      batch  160/160  loss=0.12655  elapsed=251.8s  ETA=0.0s
     8 |     0.17790 |     0.16917 |   7.81e-07 |    252.6  GPU mem  alloc=19MB  reserved=25MB
  Early stop epoch 8  (best val=0.16860)

  Total: 2022.6s  |  Best val=0.16860

  [Test — NORMALISED scale (matches TSLib / PatchTST tables)]
    MSE     : 0.1214
    MAE     : 0.2539
    RMSE    : 0.3583

  [Test — original scale]
    MSE     : 0.0011
    MAE     : 0.0211
    RMSE    : 0.0329
    MAPE%   : 2.9019

========================================================================
  VQF-F (broadcast) | Exchange Rate | pred_len=192 | enc_in=8 | device=cuda:0
  n_qubits=6 | depth=2 | batch=32 | AMP=True
========================================================================
    [train]  rows: 0..5311  (5311 total)  channels: 8
    [val]  rows: 5215..6071  (856 total)  channels: 8
    [test]  rows: 5975..7588  (1613 total)  channels: 8
  Windows → train:5024  val:569  test:1326
  [Quantum] default.qubit (6 qubits, broadcasted)  [FASTEST for small circuits]
  Params → Q(CPU):192  C(GPU):136720
  Quantum calls per batch: 8  (was 256 in original)
  GPU mem  alloc=19MB  reserved=25MB

    Ep |   Train MSE |     Val MSE |         LR |  Time(s)
  --------------------------------------------------------
    [Epoch 1] training 157 batches...
      batch    1/157  loss=0.33802  elapsed=1.8s  ETA=273.1s
      batch   15/157  loss=0.30154  elapsed=23.9s  ETA=225.8s
      batch   30/157  loss=0.23906  elapsed=47.6s  ETA=201.4s
      batch   45/157  loss=0.38949  elapsed=71.3s  ETA=177.5s
      batch   60/157  loss=0.35450  elapsed=94.9s  ETA=153.5s
      batch   75/157  loss=0.29562  elapsed=118.7s  ETA=129.8s
      batch   90/157  loss=0.40275  elapsed=142.3s  ETA=106.0s
      batch  105/157  loss=0.35901  elapsed=166.1s  ETA=82.3s
      batch  120/157  loss=0.20776  elapsed=189.7s  ETA=58.5s
      batch  135/157  loss=0.29616  elapsed=213.4s  ETA=34.8s
      batch  150/157  loss=0.27739  elapsed=237.0s  ETA=11.1s
      batch  157/157  loss=0.34531  elapsed=248.1s  ETA=0.0s
     1 |     0.32631 |     0.28884 |   1.00e-04 |    248.9  GPU mem  alloc=20MB  reserved=27MB
    [Epoch 2] training 157 batches...
      batch    1/157  loss=0.32710  elapsed=1.7s  ETA=261.4s
      batch   15/157  loss=0.22972  elapsed=23.7s  ETA=224.0s
      batch   30/157  loss=0.34709  elapsed=47.2s  ETA=199.7s
      batch   45/157  loss=0.26805  elapsed=70.8s  ETA=176.3s
      batch   60/157  loss=0.25327  elapsed=94.4s  ETA=152.6s
      batch   75/157  loss=0.35582  elapsed=118.0s  ETA=129.1s
      batch   90/157  loss=0.29640  elapsed=141.8s  ETA=105.5s
      batch  105/157  loss=0.28073  elapsed=165.4s  ETA=81.9s
      batch  120/157  loss=0.38140  elapsed=189.0s  ETA=58.3s
      batch  135/157  loss=0.34457  elapsed=212.7s  ETA=34.7s
      batch  150/157  loss=0.33917  elapsed=236.2s  ETA=11.0s
      batch  157/157  loss=0.31551  elapsed=247.4s  ETA=0.0s
     2 |     0.31117 |     0.27869 |   5.00e-05 |    248.2  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 3] training 157 batches...
      batch    1/157  loss=0.25350  elapsed=1.5s  ETA=238.1s
      batch   15/157  loss=0.39565  elapsed=23.5s  ETA=222.4s
      batch   30/157  loss=0.22778  elapsed=47.1s  ETA=199.4s
      batch   45/157  loss=0.26305  elapsed=70.7s  ETA=175.9s
      batch   60/157  loss=0.18491  elapsed=94.4s  ETA=152.6s
      batch   75/157  loss=0.24457  elapsed=118.0s  ETA=129.0s
      batch   90/157  loss=0.30937  elapsed=141.8s  ETA=105.5s
      batch  105/157  loss=0.27447  elapsed=165.4s  ETA=81.9s
      batch  120/157  loss=0.28560  elapsed=189.1s  ETA=58.3s
      batch  135/157  loss=0.23937  elapsed=212.7s  ETA=34.7s
      batch  150/157  loss=0.30031  elapsed=236.4s  ETA=11.0s
      batch  157/157  loss=0.31667  elapsed=247.4s  ETA=0.0s
     3 |     0.30397 |     0.27395 |   2.50e-05 |    248.2  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 4] training 157 batches...
      batch    1/157  loss=0.34367  elapsed=1.5s  ETA=238.9s
      batch   15/157  loss=0.25771  elapsed=23.5s  ETA=222.1s
      batch   30/157  loss=0.26839  elapsed=47.0s  ETA=199.0s
      batch   45/157  loss=0.23266  elapsed=70.5s  ETA=175.5s
      batch   60/157  loss=0.25121  elapsed=94.2s  ETA=152.3s
      batch   75/157  loss=0.28331  elapsed=117.7s  ETA=128.7s
      batch   90/157  loss=0.44176  elapsed=141.3s  ETA=105.2s
      batch  105/157  loss=0.23256  elapsed=164.9s  ETA=81.7s
      batch  120/157  loss=0.28978  elapsed=188.4s  ETA=58.1s
      batch  135/157  loss=0.26046  elapsed=212.0s  ETA=34.6s
      batch  150/157  loss=0.22098  elapsed=235.5s  ETA=11.0s
      batch  157/157  loss=0.32380  elapsed=246.5s  ETA=0.0s
     4 |     0.30084 |     0.27180 |   1.25e-05 |    247.3  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 5] training 157 batches...
      batch    1/157  loss=0.24334  elapsed=1.7s  ETA=258.5s
      batch   15/157  loss=0.31339  elapsed=23.7s  ETA=223.9s
      batch   30/157  loss=0.38356  elapsed=47.2s  ETA=199.7s
      batch   45/157  loss=0.28260  elapsed=70.8s  ETA=176.3s
      batch   60/157  loss=0.26443  elapsed=94.4s  ETA=152.6s
      batch   75/157  loss=0.30908  elapsed=118.1s  ETA=129.1s
      batch   90/157  loss=0.21924  elapsed=141.7s  ETA=105.5s
      batch  105/157  loss=0.30783  elapsed=165.4s  ETA=81.9s
      batch  120/157  loss=0.31849  elapsed=189.0s  ETA=58.3s
      batch  135/157  loss=0.25022  elapsed=212.7s  ETA=34.7s
      batch  150/157  loss=0.20679  elapsed=236.3s  ETA=11.0s
      batch  157/157  loss=0.25288  elapsed=247.3s  ETA=0.0s
     5 |     0.29940 |     0.27070 |   6.25e-06 |    248.0  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 6] training 157 batches...
      batch    1/157  loss=0.28647  elapsed=1.6s  ETA=257.0s
      batch   15/157  loss=0.21609  elapsed=23.7s  ETA=224.8s
      batch   30/157  loss=0.23430  elapsed=47.3s  ETA=200.1s
      batch   45/157  loss=0.30617  elapsed=70.9s  ETA=176.5s
      batch   60/157  loss=0.38613  elapsed=94.4s  ETA=152.6s
      batch   75/157  loss=0.27574  elapsed=118.0s  ETA=129.1s
      batch   90/157  loss=0.29692  elapsed=141.6s  ETA=105.4s
      batch  105/157  loss=0.24005  elapsed=165.2s  ETA=81.8s
      batch  120/157  loss=0.31917  elapsed=188.8s  ETA=58.2s
      batch  135/157  loss=0.21749  elapsed=212.4s  ETA=34.6s
      batch  150/157  loss=0.24424  elapsed=236.0s  ETA=11.0s
      batch  157/157  loss=0.30566  elapsed=246.9s  ETA=0.0s
     6 |     0.29871 |     0.26994 |   3.13e-06 |    247.7  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 7] training 157 batches...
      batch    1/157  loss=0.21572  elapsed=1.6s  ETA=255.7s
      batch   15/157  loss=0.27984  elapsed=23.6s  ETA=223.7s
      batch   30/157  loss=0.28482  elapsed=47.1s  ETA=199.6s
      batch   45/157  loss=0.26338  elapsed=70.8s  ETA=176.3s
      batch   60/157  loss=0.37547  elapsed=94.4s  ETA=152.6s
      batch   75/157  loss=0.26516  elapsed=118.1s  ETA=129.1s
      batch   90/157  loss=0.31798  elapsed=141.8s  ETA=105.6s
      batch  105/157  loss=0.23249  elapsed=165.4s  ETA=81.9s
      batch  120/157  loss=0.20201  elapsed=189.2s  ETA=58.3s
      batch  135/157  loss=0.24894  elapsed=212.9s  ETA=34.7s
      batch  150/157  loss=0.27588  elapsed=236.6s  ETA=11.0s
      batch  157/157  loss=0.38091  elapsed=247.6s  ETA=0.0s
     7 |     0.29839 |     0.26909 |   1.56e-06 |    248.3  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 8] training 157 batches...
      batch    1/157  loss=0.26235  elapsed=1.5s  ETA=238.5s
      batch   15/157  loss=0.25738  elapsed=23.5s  ETA=222.5s
      batch   30/157  loss=0.25107  elapsed=47.1s  ETA=199.3s
      batch   45/157  loss=0.29230  elapsed=70.6s  ETA=175.6s
      batch   60/157  loss=0.23606  elapsed=94.2s  ETA=152.3s
      batch   75/157  loss=0.29946  elapsed=117.7s  ETA=128.6s
      batch   90/157  loss=0.23026  elapsed=141.3s  ETA=105.2s
      batch  105/157  loss=0.25597  elapsed=164.9s  ETA=81.7s
      batch  120/157  loss=0.37711  elapsed=188.5s  ETA=58.1s
      batch  135/157  loss=0.27141  elapsed=212.0s  ETA=34.6s
      batch  150/157  loss=0.25404  elapsed=235.7s  ETA=11.0s
      batch  157/157  loss=0.23697  elapsed=246.7s  ETA=0.0s
     8 |     0.29823 |     0.26969 |   7.81e-07 |    247.4  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 9] training 157 batches...
      batch    1/157  loss=0.29216  elapsed=1.6s  ETA=255.3s
      batch   15/157  loss=0.22798  elapsed=23.6s  ETA=223.2s
      batch   30/157  loss=0.23590  elapsed=47.1s  ETA=199.3s
      batch   45/157  loss=0.27028  elapsed=70.8s  ETA=176.2s
      batch   60/157  loss=0.32957  elapsed=94.4s  ETA=152.6s
      batch   75/157  loss=0.24958  elapsed=118.1s  ETA=129.1s
      batch   90/157  loss=0.33474  elapsed=141.7s  ETA=105.5s
      batch  105/157  loss=0.38466  elapsed=165.5s  ETA=81.9s
      batch  120/157  loss=0.36471  elapsed=189.1s  ETA=58.3s
      batch  135/157  loss=0.21677  elapsed=212.8s  ETA=34.7s
      batch  150/157  loss=0.35216  elapsed=236.4s  ETA=11.0s
      batch  157/157  loss=0.27766  elapsed=247.5s  ETA=0.0s
     9 |     0.29814 |     0.26974 |   3.91e-07 |    248.2  GPU mem  alloc=21MB  reserved=27MB
    [Epoch 10] training 157 batches...
      batch    1/157  loss=0.25112  elapsed=1.5s  ETA=239.5s
      batch   15/157  loss=0.41990  elapsed=23.5s  ETA=222.6s
      batch   30/157  loss=0.21323  elapsed=47.0s  ETA=199.1s
      batch   45/157  loss=0.37261  elapsed=70.7s  ETA=175.9s
      batch   60/157  loss=0.28713  elapsed=94.3s  ETA=152.4s
      batch   75/157  loss=0.23048  elapsed=117.8s  ETA=128.8s
      batch   90/157  loss=0.28030  elapsed=141.4s  ETA=105.3s
      batch  105/157  loss=0.33356  elapsed=164.9s  ETA=81.7s
      batch  120/157  loss=0.22702  elapsed=188.6s  ETA=58.1s
      batch  135/157  loss=0.32070  elapsed=212.1s  ETA=34.6s
      batch  150/157  loss=0.20986  elapsed=235.8s  ETA=11.0s
      batch  157/157  loss=0.33920  elapsed=246.8s  ETA=0.0s
    10 |     0.29807 |     0.27013 |   1.95e-07 |    247.5  GPU mem  alloc=21MB  reserved=27MB
  Early stop epoch 10  (best val=0.26909)

  Total: 2479.7s  |  Best val=0.26909

  [Test — NORMALISED scale (matches TSLib / PatchTST tables)]
    MSE     : 0.2189
    MAE     : 0.3425
    RMSE    : 0.4785

  [Test — original scale]
    MSE     : 0.0019
    MAE     : 0.0285
    RMSE    : 0.0440
    MAPE%   : 3.9506

========================================================================
  VQF-F (broadcast) | Exchange Rate | pred_len=336 | enc_in=8 | device=cuda:0
  n_qubits=6 | depth=2 | batch=32 | AMP=True
========================================================================
    [train]  rows: 0..5311  (5311 total)  channels: 8
    [val]  rows: 5215..6071  (856 total)  channels: 8
    [test]  rows: 5975..7588  (1613 total)  channels: 8
  Windows → train:4880  val:425  test:1182
  [Quantum] default.qubit (6 qubits, broadcasted)  [FASTEST for small circuits]
  Params → Q(CPU):192  C(GPU):239248
  Quantum calls per batch: 8  (was 256 in original)
  GPU mem  alloc=19MB  reserved=27MB

    Ep |   Train MSE |     Val MSE |         LR |  Time(s)
  --------------------------------------------------------
    [Epoch 1] training 152 batches...
      batch    1/152  loss=0.48089  elapsed=1.6s  ETA=245.5s
      batch   15/152  loss=0.64135  elapsed=23.8s  ETA=217.3s
      batch   30/152  loss=0.35458  elapsed=47.6s  ETA=193.7s
      batch   45/152  loss=0.54063  elapsed=71.3s  ETA=169.6s
      batch   60/152  loss=0.42717  elapsed=95.1s  ETA=145.9s
      batch   75/152  loss=0.54062  elapsed=118.9s  ETA=122.0s
      batch   90/152  loss=0.41239  elapsed=142.7s  ETA=98.3s
      batch  105/152  loss=0.49022  elapsed=166.4s  ETA=74.5s
      batch  120/152  loss=0.45987  elapsed=190.2s  ETA=50.7s
      batch  135/152  loss=0.55977  elapsed=214.0s  ETA=26.9s
      batch  150/152  loss=0.38433  elapsed=237.8s  ETA=3.2s
      batch  152/152  loss=0.43155  elapsed=241.0s  ETA=0.0s
     1 |     0.49581 |     0.47375 |   1.00e-04 |    241.6  GPU mem  alloc=22MB  reserved=29MB
    [Epoch 2] training 152 batches...
      batch    1/152  loss=0.43421  elapsed=1.5s  ETA=232.2s
      batch   15/152  loss=0.41774  elapsed=23.7s  ETA=216.5s
      batch   30/152  loss=0.45610  elapsed=47.5s  ETA=193.3s
      batch   45/152  loss=0.42128  elapsed=71.2s  ETA=169.4s
      batch   60/152  loss=0.50574  elapsed=95.1s  ETA=145.7s
      batch   75/152  loss=0.36919  elapsed=118.7s  ETA=121.9s
      batch   90/152  loss=0.44896  elapsed=142.5s  ETA=98.2s
      batch  105/152  loss=0.43429  elapsed=166.2s  ETA=74.4s
      batch  120/152  loss=0.41839  elapsed=190.0s  ETA=50.7s
      batch  135/152  loss=0.55594  elapsed=213.7s  ETA=26.9s
      batch  150/152  loss=0.49335  elapsed=237.5s  ETA=3.2s
      batch  152/152  loss=0.52466  elapsed=240.6s  ETA=0.0s
     2 |     0.48282 |     0.45732 |   5.00e-05 |    241.2  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 3] training 152 batches...
      batch    1/152  loss=0.46634  elapsed=1.5s  ETA=233.6s
      batch   15/152  loss=0.32259  elapsed=23.7s  ETA=216.2s
      batch   30/152  loss=0.39997  elapsed=47.4s  ETA=192.9s
      batch   45/152  loss=0.48597  elapsed=71.1s  ETA=169.1s
      batch   60/152  loss=0.44298  elapsed=94.9s  ETA=145.5s
      batch   75/152  loss=0.37011  elapsed=118.5s  ETA=121.7s
      batch   90/152  loss=0.44870  elapsed=142.3s  ETA=98.0s
      batch  105/152  loss=0.43001  elapsed=166.0s  ETA=74.3s
      batch  120/152  loss=0.47314  elapsed=189.8s  ETA=50.6s
      batch  135/152  loss=0.51118  elapsed=213.6s  ETA=26.9s
      batch  150/152  loss=0.52073  elapsed=237.3s  ETA=3.2s
      batch  152/152  loss=0.40241  elapsed=240.5s  ETA=0.0s
     3 |     0.47499 |     0.44814 |   2.50e-05 |    241.0  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 4] training 152 batches...
      batch    1/152  loss=0.39489  elapsed=1.5s  ETA=233.3s
      batch   15/152  loss=0.42777  elapsed=23.8s  ETA=217.3s
      batch   30/152  loss=0.48385  elapsed=47.5s  ETA=193.0s
      batch   45/152  loss=0.45550  elapsed=71.2s  ETA=169.4s
      batch   60/152  loss=0.53258  elapsed=94.9s  ETA=145.5s
      batch   75/152  loss=0.61438  elapsed=118.7s  ETA=121.8s
      batch   90/152  loss=0.65490  elapsed=142.4s  ETA=98.1s
      batch  105/152  loss=0.52701  elapsed=166.2s  ETA=74.4s
      batch  120/152  loss=0.54883  elapsed=189.9s  ETA=50.6s
      batch  135/152  loss=0.49744  elapsed=213.7s  ETA=26.9s
      batch  150/152  loss=0.41287  elapsed=237.3s  ETA=3.2s
      batch  152/152  loss=0.54217  elapsed=240.5s  ETA=0.0s
     4 |     0.47206 |     0.44659 |   1.25e-05 |    241.1  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 5] training 152 batches...
      batch    1/152  loss=0.48226  elapsed=1.6s  ETA=248.4s
      batch   15/152  loss=0.49319  elapsed=23.7s  ETA=216.4s
      batch   30/152  loss=0.52013  elapsed=47.3s  ETA=192.3s
      batch   45/152  loss=0.47013  elapsed=71.0s  ETA=168.8s
      batch   60/152  loss=0.52162  elapsed=94.5s  ETA=145.0s
      batch   75/152  loss=0.45806  elapsed=118.3s  ETA=121.4s
      batch   90/152  loss=0.54821  elapsed=141.9s  ETA=97.7s
      batch  105/152  loss=0.46246  elapsed=165.6s  ETA=74.1s
      batch  120/152  loss=0.68296  elapsed=189.2s  ETA=50.5s
      batch  135/152  loss=0.63325  elapsed=213.0s  ETA=26.8s
      batch  150/152  loss=0.42687  elapsed=236.6s  ETA=3.2s
      batch  152/152  loss=0.33793  elapsed=239.8s  ETA=0.0s
     5 |     0.47133 |     0.44430 |   6.25e-06 |    240.4  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 6] training 152 batches...
      batch    1/152  loss=0.45972  elapsed=1.6s  ETA=248.4s
      batch   15/152  loss=0.43587  elapsed=23.7s  ETA=216.8s
      batch   30/152  loss=0.41149  elapsed=47.5s  ETA=193.1s
      batch   45/152  loss=0.48229  elapsed=71.2s  ETA=169.3s
      batch   60/152  loss=0.42275  elapsed=94.9s  ETA=145.5s
      batch   75/152  loss=0.48523  elapsed=118.7s  ETA=121.9s
      batch   90/152  loss=0.50557  elapsed=142.4s  ETA=98.1s
      batch  105/152  loss=0.55365  elapsed=166.2s  ETA=74.4s
      batch  120/152  loss=0.48725  elapsed=190.0s  ETA=50.7s
      batch  135/152  loss=0.44509  elapsed=213.7s  ETA=26.9s
      batch  150/152  loss=0.41173  elapsed=237.5s  ETA=3.2s
      batch  152/152  loss=0.54573  elapsed=240.5s  ETA=0.0s
     6 |     0.46990 |     0.44359 |   3.13e-06 |    241.2  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 7] training 152 batches...
      batch    1/152  loss=0.46513  elapsed=1.5s  ETA=233.1s
      batch   15/152  loss=0.46093  elapsed=23.7s  ETA=216.4s
      batch   30/152  loss=0.56052  elapsed=47.4s  ETA=192.7s
      batch   45/152  loss=0.44431  elapsed=71.2s  ETA=169.2s
      batch   60/152  loss=0.36320  elapsed=94.9s  ETA=145.6s
      batch   75/152  loss=0.51534  elapsed=118.6s  ETA=121.7s
      batch   90/152  loss=0.42284  elapsed=142.3s  ETA=98.1s
      batch  105/152  loss=0.48797  elapsed=166.0s  ETA=74.3s
      batch  120/152  loss=0.53642  elapsed=189.8s  ETA=50.6s
      batch  135/152  loss=0.43226  elapsed=213.5s  ETA=26.9s
      batch  150/152  loss=0.36471  elapsed=237.2s  ETA=3.2s
      batch  152/152  loss=0.45199  elapsed=240.3s  ETA=0.0s
     7 |     0.46990 |     0.44273 |   1.56e-06 |    240.9  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 8] training 152 batches...
      batch    1/152  loss=0.57466  elapsed=1.5s  ETA=232.9s
      batch   15/152  loss=0.55052  elapsed=23.6s  ETA=215.7s
      batch   30/152  loss=0.46137  elapsed=47.4s  ETA=192.6s
      batch   45/152  loss=0.39824  elapsed=71.0s  ETA=168.8s
      batch   60/152  loss=0.48031  elapsed=94.8s  ETA=145.4s
      batch   75/152  loss=0.50004  elapsed=118.4s  ETA=121.6s
      batch   90/152  loss=0.42460  elapsed=142.3s  ETA=98.0s
      batch  105/152  loss=0.61222  elapsed=166.0s  ETA=74.3s
      batch  120/152  loss=0.42792  elapsed=189.8s  ETA=50.6s
      batch  135/152  loss=0.50521  elapsed=213.5s  ETA=26.9s
      batch  150/152  loss=0.41352  elapsed=237.2s  ETA=3.2s
      batch  152/152  loss=0.37250  elapsed=240.4s  ETA=0.0s
     8 |     0.46994 |     0.44314 |   7.81e-07 |    241.0  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 9] training 152 batches...
      batch    1/152  loss=0.48714  elapsed=1.5s  ETA=232.9s
      batch   15/152  loss=0.40948  elapsed=23.8s  ETA=217.4s
      batch   30/152  loss=0.45774  elapsed=47.5s  ETA=193.2s
      batch   45/152  loss=0.47693  elapsed=71.3s  ETA=169.6s
      batch   60/152  loss=0.45864  elapsed=95.0s  ETA=145.7s
      batch   75/152  loss=0.48290  elapsed=118.8s  ETA=122.0s
      batch   90/152  loss=0.51404  elapsed=142.5s  ETA=98.2s
      batch  105/152  loss=0.47983  elapsed=166.3s  ETA=74.4s
      batch  120/152  loss=0.50395  elapsed=190.0s  ETA=50.7s
      batch  135/152  loss=0.47350  elapsed=213.8s  ETA=26.9s
      batch  150/152  loss=0.49973  elapsed=237.4s  ETA=3.2s
      batch  152/152  loss=0.63605  elapsed=240.6s  ETA=0.0s
     9 |     0.46983 |     0.44300 |   3.91e-07 |    241.2  GPU mem  alloc=23MB  reserved=31MB
    [Epoch 10] training 152 batches...
      batch    1/152  loss=0.51156  elapsed=1.6s  ETA=248.9s
      batch   15/152  loss=0.47873  elapsed=23.8s  ETA=217.3s
      batch   30/152  loss=0.45036  elapsed=47.4s  ETA=192.9s
      batch   45/152  loss=0.37181  elapsed=71.1s  ETA=169.0s
      batch   60/152  loss=0.48814  elapsed=94.8s  ETA=145.4s
      batch   75/152  loss=0.50598  elapsed=118.6s  ETA=121.8s
      batch   90/152  loss=0.42528  elapsed=142.2s  ETA=98.0s
      batch  105/152  loss=0.43852  elapsed=166.0s  ETA=74.3s
      batch  120/152  loss=0.55118  elapsed=189.7s  ETA=50.6s
      batch  135/152  loss=0.44835  elapsed=213.5s  ETA=26.9s
      batch  150/152  loss=0.51846  elapsed=237.2s  ETA=3.2s
      batch  152/152  loss=0.48284  elapsed=240.3s  ETA=0.0s
    10 |     0.47038 |     0.44226 |   1.95e-07 |    240.9  GPU mem  alloc=23MB  reserved=31MB

  Total: 2410.5s  |  Best val=0.44226

  [Test — NORMALISED scale (matches TSLib / PatchTST tables)]
    MSE     : 0.3757
    MAE     : 0.4541
    RMSE    : 0.6250

  [Test — original scale]
    MSE     : 0.0033
    MAE     : 0.0375
    RMSE    : 0.0577
    MAPE%   : 5.3022

========================================================================
  VQF-F (broadcast) | Exchange Rate | pred_len=720 | enc_in=8 | device=cuda:0
  n_qubits=6 | depth=2 | batch=32 | AMP=True
========================================================================
    [train]  rows: 0..5311  (5311 total)  channels: 8
    [val]  rows: 5215..6071  (856 total)  channels: 8
    [test]  rows: 5975..7588  (1613 total)  channels: 8
  Windows → train:4496  val:41  test:798
  [Quantum] default.qubit (6 qubits, broadcasted)  [FASTEST for small circuits]
  Params → Q(CPU):192  C(GPU):512656
  Quantum calls per batch: 8  (was 256 in original)
  GPU mem  alloc=20MB  reserved=31MB

    Ep |   Train MSE |     Val MSE |         LR |  Time(s)
  --------------------------------------------------------
    [Epoch 1] training 140 batches...
      batch    1/140  loss=1.09204  elapsed=1.8s  ETA=244.7s
      batch   14/140  loss=0.88754  elapsed=22.2s  ETA=200.2s
      batch   28/140  loss=0.83343  elapsed=44.3s  ETA=177.4s
      batch   42/140  loss=0.91180  elapsed=66.5s  ETA=155.1s
      batch   56/140  loss=0.89507  elapsed=88.5s  ETA=132.8s
      batch   70/140  loss=0.71658  elapsed=110.6s  ETA=110.6s
      batch   84/140  loss=0.87816  elapsed=132.8s  ETA=88.5s
      batch   98/140  loss=0.86184  elapsed=154.9s  ETA=66.4s
      batch  112/140  loss=0.91516  elapsed=177.0s  ETA=44.2s
      batch  126/140  loss=0.74579  elapsed=199.2s  ETA=22.1s
      batch  140/140  loss=1.07808  elapsed=221.3s  ETA=0.0s
     1 |     0.87670 |     1.22609 |   1.00e-04 |    221.4  GPU mem  alloc=26MB  reserved=52MB
    [Epoch 2] training 140 batches...
      batch    1/140  loss=0.76667  elapsed=1.5s  ETA=214.6s
      batch   14/140  loss=0.81274  elapsed=22.3s  ETA=200.3s
      batch   28/140  loss=0.87381  elapsed=44.5s  ETA=177.8s
      batch   42/140  loss=0.89936  elapsed=66.7s  ETA=155.6s
      batch   56/140  loss=0.76869  elapsed=88.9s  ETA=133.4s
      batch   70/140  loss=0.75325  elapsed=111.1s  ETA=111.1s
      batch   84/140  loss=0.86881  elapsed=133.5s  ETA=89.0s
      batch   98/140  loss=0.80433  elapsed=155.5s  ETA=66.7s
      batch  112/140  loss=0.79946  elapsed=177.8s  ETA=44.4s
      batch  126/140  loss=0.80454  elapsed=200.0s  ETA=22.2s
      batch  140/140  loss=0.80040  elapsed=222.1s  ETA=0.0s
     2 |     0.86324 |     1.20868 |   5.00e-05 |    222.2  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 3] training 140 batches...
      batch    1/140  loss=0.91658  elapsed=1.5s  ETA=213.9s
      batch   14/140  loss=0.89529  elapsed=22.1s  ETA=198.9s
      batch   28/140  loss=0.87610  elapsed=44.2s  ETA=176.9s
      batch   42/140  loss=0.93109  elapsed=66.3s  ETA=154.8s
      batch   56/140  loss=0.90338  elapsed=88.4s  ETA=132.7s
      batch   70/140  loss=0.89300  elapsed=110.5s  ETA=110.5s
      batch   84/140  loss=0.74566  elapsed=132.6s  ETA=88.4s
      batch   98/140  loss=0.82869  elapsed=154.7s  ETA=66.3s
      batch  112/140  loss=0.83436  elapsed=176.8s  ETA=44.2s
      batch  126/140  loss=0.83867  elapsed=198.9s  ETA=22.1s
      batch  140/140  loss=0.92308  elapsed=221.0s  ETA=0.0s
     3 |     0.85806 |     1.20404 |   2.50e-05 |    221.1  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 4] training 140 batches...
      batch    1/140  loss=1.03307  elapsed=1.5s  ETA=215.0s
      batch   14/140  loss=0.89931  elapsed=22.2s  ETA=200.0s
      batch   28/140  loss=0.90771  elapsed=44.4s  ETA=177.7s
      batch   42/140  loss=0.82418  elapsed=66.6s  ETA=155.4s
      batch   56/140  loss=0.75683  elapsed=88.8s  ETA=133.2s
      batch   70/140  loss=0.74740  elapsed=111.0s  ETA=111.0s
      batch   84/140  loss=0.77479  elapsed=133.3s  ETA=88.9s
      batch   98/140  loss=0.88746  elapsed=155.5s  ETA=66.6s
      batch  112/140  loss=0.85063  elapsed=177.7s  ETA=44.4s
      batch  126/140  loss=0.80965  elapsed=200.0s  ETA=22.2s
      batch  140/140  loss=0.98249  elapsed=222.1s  ETA=0.0s
     4 |     0.85474 |     1.19843 |   1.25e-05 |    222.2  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 5] training 140 batches...
      batch    1/140  loss=0.84012  elapsed=1.7s  ETA=229.7s
      batch   14/140  loss=0.96279  elapsed=22.3s  ETA=200.7s
      batch   28/140  loss=0.84373  elapsed=44.5s  ETA=177.9s
      batch   42/140  loss=0.87719  elapsed=66.7s  ETA=155.5s
      batch   56/140  loss=0.80862  elapsed=88.9s  ETA=133.3s
      batch   70/140  loss=0.94298  elapsed=111.1s  ETA=111.1s
      batch   84/140  loss=0.76348  elapsed=133.4s  ETA=88.9s
      batch   98/140  loss=0.73181  elapsed=155.6s  ETA=66.7s
      batch  112/140  loss=0.81728  elapsed=177.8s  ETA=44.5s
      batch  126/140  loss=0.74482  elapsed=200.0s  ETA=22.2s
      batch  140/140  loss=0.83869  elapsed=222.2s  ETA=0.0s
     5 |     0.85385 |     1.19906 |   6.25e-06 |    222.2  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 6] training 140 batches...
      batch    1/140  loss=0.82200  elapsed=1.7s  ETA=230.9s
      batch   14/140  loss=0.93569  elapsed=22.2s  ETA=200.2s
      batch   28/140  loss=0.94125  elapsed=44.5s  ETA=177.9s
      batch   42/140  loss=0.73045  elapsed=66.7s  ETA=155.7s
      batch   56/140  loss=0.81041  elapsed=89.1s  ETA=133.6s
      batch   70/140  loss=0.70149  elapsed=111.3s  ETA=111.3s
      batch   84/140  loss=0.92459  elapsed=133.5s  ETA=89.0s
      batch   98/140  loss=0.80721  elapsed=155.7s  ETA=66.7s
      batch  112/140  loss=0.90832  elapsed=177.9s  ETA=44.5s
      batch  126/140  loss=0.69662  elapsed=200.2s  ETA=22.2s
      batch  140/140  loss=0.82977  elapsed=222.3s  ETA=0.0s
     6 |     0.85261 |     1.19623 |   3.13e-06 |    222.4  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 7] training 140 batches...
      batch    1/140  loss=0.86123  elapsed=1.5s  ETA=215.1s
      batch   14/140  loss=0.84575  elapsed=22.1s  ETA=199.3s
      batch   28/140  loss=0.95169  elapsed=44.3s  ETA=177.2s
      batch   42/140  loss=0.75330  elapsed=66.5s  ETA=155.2s
      batch   56/140  loss=0.90473  elapsed=88.9s  ETA=133.3s
      batch   70/140  loss=0.90597  elapsed=111.1s  ETA=111.1s
      batch   84/140  loss=0.79783  elapsed=133.4s  ETA=88.9s
      batch   98/140  loss=0.78393  elapsed=155.6s  ETA=66.7s
      batch  112/140  loss=0.90232  elapsed=177.8s  ETA=44.5s
      batch  126/140  loss=0.86209  elapsed=200.1s  ETA=22.2s
      batch  140/140  loss=0.86048  elapsed=222.2s  ETA=0.0s
     7 |     0.85215 |     1.19291 |   1.56e-06 |    222.3  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 8] training 140 batches...
      batch    1/140  loss=0.94472  elapsed=1.5s  ETA=214.6s
      batch   14/140  loss=0.79116  elapsed=22.2s  ETA=199.7s
      batch   28/140  loss=0.95291  elapsed=44.4s  ETA=177.6s
      batch   42/140  loss=0.91554  elapsed=66.6s  ETA=155.4s
      batch   56/140  loss=0.88276  elapsed=88.8s  ETA=133.2s
      batch   70/140  loss=0.81289  elapsed=111.0s  ETA=111.0s
      batch   84/140  loss=0.86155  elapsed=133.3s  ETA=88.9s
      batch   98/140  loss=0.80187  elapsed=155.5s  ETA=66.6s
      batch  112/140  loss=0.88323  elapsed=177.6s  ETA=44.4s
      batch  126/140  loss=0.97260  elapsed=199.8s  ETA=22.2s
      batch  140/140  loss=0.96013  elapsed=221.9s  ETA=0.0s
     8 |     0.85180 |     1.19501 |   7.81e-07 |    222.0  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 9] training 140 batches...
      batch    1/140  loss=0.95029  elapsed=1.5s  ETA=214.7s
      batch   14/140  loss=0.98147  elapsed=22.2s  ETA=199.9s
      batch   28/140  loss=0.79359  elapsed=44.4s  ETA=177.6s
      batch   42/140  loss=0.74973  elapsed=66.6s  ETA=155.4s
      batch   56/140  loss=0.86093  elapsed=88.8s  ETA=133.2s
      batch   70/140  loss=0.78975  elapsed=111.0s  ETA=111.0s
      batch   84/140  loss=0.83956  elapsed=133.4s  ETA=88.9s
      batch   98/140  loss=0.98757  elapsed=155.5s  ETA=66.7s
      batch  112/140  loss=0.75883  elapsed=177.7s  ETA=44.4s
      batch  126/140  loss=0.82264  elapsed=199.9s  ETA=22.2s
      batch  140/140  loss=0.83655  elapsed=222.0s  ETA=0.0s
     9 |     0.85256 |     1.19602 |   3.91e-07 |    222.1  GPU mem  alloc=28MB  reserved=55MB
    [Epoch 10] training 140 batches...
      batch    1/140  loss=0.82435  elapsed=1.5s  ETA=213.9s
      batch   14/140  loss=0.95295  elapsed=22.2s  ETA=199.4s
      batch   28/140  loss=0.88252  elapsed=44.3s  ETA=177.3s
      batch   42/140  loss=0.84487  elapsed=66.5s  ETA=155.2s
      batch   56/140  loss=0.86138  elapsed=88.7s  ETA=133.0s
      batch   70/140  loss=0.89462  elapsed=110.9s  ETA=110.9s
      batch   84/140  loss=0.84587  elapsed=133.1s  ETA=88.7s
      batch   98/140  loss=0.80555  elapsed=155.3s  ETA=66.5s
      batch  112/140  loss=0.82506  elapsed=177.6s  ETA=44.4s
      batch  126/140  loss=0.86152  elapsed=199.8s  ETA=22.2s
      batch  140/140  loss=0.89205  elapsed=222.0s  ETA=0.0s
    10 |     0.85216 |     1.19869 |   1.95e-07 |    222.0  GPU mem  alloc=28MB  reserved=55MB
  Early stop epoch 10  (best val=1.19291)

  Total: 2220.0s  |  Best val=1.19291

  [Test — NORMALISED scale (matches TSLib / PatchTST tables)]
    MSE     : 0.8532
    MAE     : 0.6999


========================================================================
      VQF-F on Exchange Rate — NORMALISED metrics (TSLib protocol)      
        (directly comparable to PatchTST / iTransformer tables)         
========================================================================
     H |   MSE (norm) |   MAE (norm) |  
------------------------------------------------------------------------
    96 |       0.1214 |       0.2539 |      
   192 |       0.2189 |       0.3425 |       
   336 |       0.3757 |       0.4541 |     
   720 |       0.8532 |       0.6999 |      
------------------------------------------------------------------------












