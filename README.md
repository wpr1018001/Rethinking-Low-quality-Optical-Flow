# [<b>Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation</b>](https://arxiv.org/abs/2403.10039)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19043-b31b1b.svg)](https://arxiv.org/abs/2403.10039)

Before releasing the training method you can try replacing the RCF counterpart file for the training


## TODO
- [ ] **Core Code Release**
  - [x] flow_aggregation_head_with_residual.py
  - [x] data.py
  - [ ] main_test.py
- [ ] **Training Method**
- [ ] **Pretrain Model**

## Citation

If you find the dataset or code useful, please cite:

```bibtex
@misc{wu2024rethinking,
      title={Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation}, 
      author={Peiran Wu and Yang Liu and Jiayu Huo and Gongyu Zhang and Christos Bergeles and Rachel Sparks and Prokar Dasgupta and Alejandro Granados and Sebastien Ourselin},
      year={2024},
      eprint={2403.10039},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
} 
```
## Acknowledgements
Our code is based on / inspired by the following repositories:
* https://github.com/TonyLianLong/RCF-UnsupVideoSeg (published under [MIT License])
