# Experiments

| N.   | Type                        | H.params                   | Samples |                    Remarks                     |                                                     Image(s) |
| :--- | --------------------------- | :------------------------- | :-----: | :--------------------------------------------: | -----------------------------------------------------------: |
| 1    | Conv1D (inception~ encoder) |                            |   All   |         No horizontal differentiation          | ![recon_4@batch_100600](/home/krishna/Documents/beatbrain/logs/cvae_1d_halfinception/png/reconstructed/recon_4@batch_100600.png) |
| 2    | Conv2D                      | 1 Conv Block; LR=1e-4      |    1    |                                                | ![recon_1@batch_4900](/home/krishna/Documents/beatbrain/logs/cvae_2d_1sample/png/reconstructed/recon_1@batch_4900.png) |
| 3    | Conv2D                      | 1 Conv Block; LR=1e-5      |    1    |                                                | ![recon_1@batch_4700](/home/krishna/Documents/beatbrain/logs/cvae_2d_1sample_2/png/reconstructed/recon_1@batch_4700.png) |
| 4    | Conv2D                      | 4 Conv Blocks; LR=1e-4     |    1    |                                                | ![recon_1@batch_1800](/home/krishna/Documents/beatbrain/logs/cvae_2d_1sample_3/png/reconstructed/recon_1@batch_1800.png) |
| 5    | Conv2D                      | 2 Conv Blocks; LR=1e-4     |    1    |                                                | ![recon_1@batch_4000](/home/krishna/Documents/beatbrain/logs/cvae_2d_1sample_4/png/reconstructed/recon_1@batch_4000.jpg) |
| 6    | Conv2D-inception            | 1 Inception Block; LR=1e-4 |    1    | Unreliable - almost alyways contains artifacts | ![recon_1@batch_3800](/home/krishna/Documents/beatbrain/logs/cvae_2d_inception_1sample/png/reconstructed/recon_1@batch_3800.png) |
| 7    | Conv2D-inception            | 1 Inception Block; LR=1e-4 |    1    |                  artifact(s)                   | ![recon_1@batch_4900](/home/krishna/Documents/beatbrain/logs/cvae_2d_inception_1sample_2/png/reconstructed/recon_1@batch_4900.png) |
| 8    | Conv2D                      | 2 Conv Blocks              |    2    |                a little blurry                 | ![recon_2@batch_8500](/home/krishna/Documents/beatbrain/logs/cvae_2d_2samples/png/reconstructed/recon_2@batch_8500.jpg) |
| 9    |                             |                            |         |                                                |                                                              |

