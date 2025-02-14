<div align="center">
<img src="./assets/Logo.png" alt="Image Alt Text" width="150" height="150">
<h3> FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models </h3>
<!-- <h4> CVPR 2024 </h4> -->
  
[Zhipei Xu](https://villa.jianzhang.tech/people/zhipei-xu-%E5%BE%90%E5%BF%97%E6%B2%9B/), [Xuanyu Zhang](https://xuanyuzhang21.github.io/), [Runyi Li](https://villa.jianzhang.tech/people/runyi-li-%E6%9D%8E%E6%B6%A6%E4%B8%80/), [Zecheng Tang](https://villa.jianzhang.tech/people/zecheng-tang-%E6%B1%A4%E6%B3%BD%E5%9F%8E/), [Qing Huang](https://github.com/zhipeixu/FakeShield), [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University



[![arXiv](https://img.shields.io/badge/Arxiv-2410.02761-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.02761) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/zhipeixu/FakeShield/blob/main/LICENSE) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhipeixu%2FFakeShield&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![hf_space](https://img.shields.io/badge/🤗-Huggingface%20Checkpoint-blue.svg)](https://huggingface.co/zhipeixu/FakeShield)
[![Home Page](https://img.shields.io/badge/Project_Page-FakeShield-blue.svg)](https://zhipeixu.github.io/projects/FakeShield/)
  <br>
[![wechat](https://img.shields.io/badge/-WeChat@新智元-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/_ih1EycGsUTYRK15X2OrRA)
[![wechat](https://img.shields.io/badge/-WeChat@52CV-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/a7WpY7TuB7V3M7r3FMxRfA)
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/3053214498)
[![csdn](https://img.shields.io/badge/-CSDN-000000?logo=CSDN&logoColor=DC143C)](https://blog.csdn.net/amusi1994/article/details/142892876)


</div>


---


<details open><summary>💡 We also have other **Copyright Protection** projects that may interest you ✨. </summary><p>
<!--  may -->

> [**EditGuard: Versatile Image Watermarking for Tamper Localization and Copyright Protection [CVPR 2024]**](https://arxiv.org/abs/2312.08883) <br>
> Xuanyu Zhang, Runyi Li, Jiwen Yu, Youmin Xu, Weiqi Li, Jian Zhang <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/xuanyuzhang21/EditGuard)  [![github](https://img.shields.io/github/stars/xuanyuzhang21/EditGuard.svg?style=social)](https://github.com/xuanyuzhang21/EditGuard) [![arXiv](https://img.shields.io/badge/Arxiv-2312.08883-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.08883) <br>

> [**V2A-Mark: Versatile Deep Visual-Audio Watermarking for Manipulation Localization and Copyright Protection [ACM MM 2024]**](https://arxiv.org/pdf/2404.16824) <br>
> Xuanyu Zhang, Youmin Xu, Runyi Li, Jiwen Yu, Weiqi Li, Zhipei Xu, Jian Zhang <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/xuanyuzhang21/EditGuard)  [![github](https://img.shields.io/github/stars/xuanyuzhang21/EditGuard.svg?style=social)](https://github.com/xuanyuzhang21/EditGuard) [![arXiv](https://img.shields.io/badge/Arxiv-2404.16824-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2404.16824) <br>

> [**GS-Hider: Hiding Messages into 3D Gaussian Splatting [NeurlPS 2024]**](https://arxiv.org/pdf/2405.15118) <br>
> Xuanyu Zhang, Jiarui Meng, Runyi Li, Zhipei Xu, Yongbing Zhang, Jian Zhang <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/xuanyuzhang21/GS-Hider)  [![github](https://img.shields.io/github/stars/xuanyuzhang21/GS-Hider.svg?style=social)](https://github.com/xuanyuzhang21/GS-Hider) [![arXiv](https://img.shields.io/badge/Arxiv-2405.15118-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2405.15118) <br>

</p></details>


## 📰 News
* **[2025.02.14]** 🤗  We are progressively open-sourcing **all code & pre-trained model weights**. Welcome to **watch** 👀 this repository for the latest updates.
* **[2025.01.23]** 🎉🎉🎉 Our FakeShield has been accepted at ICLR 2025! 
* **[2024.10.03]**  🔥 We have released **FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models**. We present explainable IFDL tasks, constructing the MMTD-Set dataset and the FakeShield framework. Check out the [paper](https://arxiv.org/abs/2410.02761). The code and dataset are coming soon


## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/128/1022/1022330.png"> FakeShield Overview

FakeShield is a novel multi-modal framework designed for explainable image forgery detection and localization (IFDL). Unlike traditional black-box IFDL methods, FakeShield integrates multi-modal large language models (MLLMs) to analyze manipulated images, generate tampered region masks, and provide human-understandable explanations based on pixel-level artifacts and semantic inconsistencies. To improve generalization across diverse forgery types, FakeShield introduces domain tags, which guide the model to recognize different manipulation techniques effectively. Additionally, we construct MMTD-Set, a richly annotated dataset containing multi-modal descriptions of manipulated images, fostering better interpretability. Through extensive experiments, FakeShield demonstrates superior performance in detecting and localizing various forgeries, including copy-move, splicing, removal, DeepFake, and AI-generated manipulations.

![alt text](assets/teasor.png)


## 🏆 Contributions

- **FakeShield Introduction.** We introduce FakeShield, a multi-modal framework for explainable image forgery detection and localization, which is **the first** to leverage MLLMs for the IFDL task. We also propose Domain Tag-guided Explainable Forgery Detection Module(DTE-FDM) and Multimodal Forgery Localization Module (MFLM) to improve the generalization and robustness of the models

- **Novel Explainable-IFDL Task.** We propose **the first** explainable image forgery detection and localization (e-IFDL) task, addressing the opacity of traditional IFDL methods by providing both pixel-level and semantic-level explanations.  

- **MMTD-Set Dataset Construction.** We create the MMTD-Set by enriching existing IFDL datasets using GPT-4o, generating high-quality “image-mask-description” triplets for enhanced multimodal learning.  

![alt text](assets/figure1-pipeline.png)

## 🛠️ Requirements and Installation

* Python == 3.9
* Pytorch == 1.13.0
* CUDA Version == 11.6

Install required packages:

```bash
apt update && apt install git
pip install -r requirements.txt

git clone https://github.com/open-mmlab/mmcv
cd mmcv
git checkout v1.4.7
MMCV_WITH_OPS=1 pip install -e .

cd ../DTE-FDM
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## 🤖 Quick Start Demo



## 🚀 Training & Validating

##  📚 Main Results

## 📜 Citation

```bibtex
    @article{xu2024fakeshield,
            title={FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models},
            author={Xu, Zhipei and Zhang, Xuanyu and Li, Runyi and Tang, Zecheng and Huang, Qing and Zhang, Jian},
            journal={International Conference on Learning Representations},
            year={2025}
    }
```

## 🙏 Acknowledgement

We are thankful to LLaVA, groundingLMM, and LISA for releasing their models and code as open-source contributions.