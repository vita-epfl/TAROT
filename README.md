<div align="center">
<img align="left" width="80" height="80" src="docs/assets/tarot.png" alt="">

# TAROT: Targeted Data Selection via Optimal Transport
[Lan Feng](https://alan-lanfeng.github.io/), [Fan Nie](https://scholar.google.com/citations?user=o2lsU8YAAAAJ&hl=en), [Yuejiang Liu](https://sites.google.com/view/yuejiangliu/home), [Alexandre Alahi](https://people.epfl.ch/alexandre.alahi?lang=en)

[Visual Intelligence for Transportation Lab, EPFL](https://www.epfl.ch/labs/vita/) 

Stanford University
</div>
<div align="center">
<img align="center" src="docs/assets/pull.png" alt="System Overview" width="500">
</div>

---

## 🛠 Environment Setup

To get started with TAROT, follow these steps:
   - Install [GeomLoss](https://www.kernel-operations.io/geomloss/).  
   - Install [Trak](https://trak.readthedocs.io/en/latest/).

---

## 🚀 Quick Start

### 🚗[Motion Prediction](./examples/motion_prediction/README.md)

### 📖[Instruction Tuning](./examples/motion_prediction/README.md)

### 🏞️[Semantic Segmentation](./examples/motion_prediction/README.md)


## 📊Qualitative Results of WFD
![system](docs/assets/inf.png)

## Acknowledgement
This repo relies on the [TRAK](https://trak.readthedocs.io/en/latest/) implementation to compute the projected gradients. We are immensely grateful to the authors of that project.

## For citation:

```
@misc{feng2024tarot,
    title={TAROT: Targeted Data Selection via Optimal Transport},
    author={Lan Feng and Fan Nie and Yuejiang Liu and Alexandre Alahi},
    year={2024},
    eprint={2412.00420},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}