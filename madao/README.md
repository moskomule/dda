# MADAO

This is an official *re*implementation of
MADAO ([hataya+22](https://openaccess.thecvf.com/content/WACV2022/papers/Hataya_Meta_Approach_to_Data_Augmentation_Optimization_WACV_2022_paper.pdf))
.

## Requirements

* `Python>=3.9`
* `PyTorch==1.11.0`
* `torchvision`
* `kornia`
* `homura-core==2021.12.1`
* `chika`

## How to Run

```
python main.py --data.name {cifar10,cifar100,svhn}
```

## Disclaimer

This implementation prioritizes easy-to-use, and thus, is not as fast as the original implementation.



## Citation

```bibtex
@InProceedings{hataya2022,
    author    = {Hataya, Ryuichiro and Zdenek, Jan and Yoshizoe, Kazuki and Nakayama, Hideki},
    title     = {Meta Approach to Data Augmentation Optimization},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {2574-2583}
}
```