# [<b>Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation</b>](https://arxiv.org/abs/2403.10039)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19043-b31b1b.svg)](https://arxiv.org/abs/2403.10039)

Before releasing the training method you can try replacing the RCF counterpart file for the training

## Datasets
You can download our processed Endovis 2017 dataset from [Google Drive](https://drive.google.com/file/d/1N44QWv4y7Cjxe2nOzXA54VlKnVYFrB3Y/view?usp=drive_link)

We used the [RAFT](https://arxiv.org/pdf/2003.12039.pdf) model inference to obtain the optical flow data.

## TODO
- [x] **Optical Flow Inference Code**
- [x] **Datasets**
- [ ] **Core Code Release**
  - [x] flow_aggregation_head_with_residual.py
  - [x] data.py
  - [ ] main_test.py
- [ ] **Training Method**
- [ ] **Pretrain Model**

## Citation

If you find the dataset or code useful, please cite:

```bibtex
@article{wu2024rethinking,
  title={Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation},
  author={Wu, Peiran and Liu, Yang and Huo, Jiayu and Zhang, Gongyu and Bergeles, Christos and Sparks, Rachel and Dasgupta, Prokar and Granados, Alejandro and Ourselin, Sebastien},
  journal={arXiv preprint arXiv:2403.10039},
  year={2024}
}
```
## Acknowledgements
Our code is based on / inspired by the following repositories:
* https://github.com/TonyLianLong/RCF-UnsupVideoSeg (published under [MIT License])
