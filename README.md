<div align="center">
<img src="./asserts/Logo.png" alt="Image Alt Text" width="150" height="150">
<h3> FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models </h3>
<!-- <h4> CVPR 2024 </h4> -->
  
[Zhipei Xu](https://villa.jianzhang.tech/people/zhipei-xu-%E5%BE%90%E5%BF%97%E6%B2%9B/), [Xuanyu Zhang](https://xuanyuzhang21.github.io/), [Runyi Li](https://villa.jianzhang.tech/people/runyi-li-%E6%9D%8E%E6%B6%A6%E4%B8%80/), [Zecheng Tang](https://villa.jianzhang.tech/people/zecheng-tang-%E6%B1%A4%E6%B3%BD%E5%9F%8E/), [Qing Huang](https://github.com/zhipeixu/FakeShield), [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University

[![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)](https://arxiv.org/abs/2410.02761)
[![Home Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://zhipeixu.github.io/projects/FakeShield/)
[![zhihu](https://img.shields.io/badge/-WeChat@新智元-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/_ih1EycGsUTYRK15X2OrRA)

</div>

## News
- 🔥 We have released **FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models**. We present explainable IFDL tasks, constructing the MMTD-Set dataset and the FakeShield framework. Check out the [paper](https://arxiv.org/abs/2410.02761). The code and dataset are coming soon

## Abstract
The rapid development of generative AI is a double-edged sword, which not only facilitates content creation but also makes image manipulation easier and more difficult to detect. Although current image forgery detection and localization (IFDL) methods are generally effective, they tend to face two challenges: 1) black-box nature with unknown detection principle, 2) limited generalization across diverse tampering methods (e.g., Photoshop, DeepFake, AIGC-Editing). To address these issues, we propose the explainable IFDL task and design FakeShield, a multi-modal framework capable of evaluating image authenticity, generating tampered region masks, and providing a judgment basis based on pixel-level and image-level tampering clues. Additionally, we leverage GPT-4o to enhance existing IFDL datasets, creating the Multi-Modal Tamper Description dataSet (MMTD-Set) for training FakeShield’s tampering analysis capabilities. Meanwhile, we incorporate a Domain Tag-guided Explainable Forgery Detection Module (DTE-FDM) and a Multi-modal Forgery Localization Module (MFLM) to address various types of tamper detection interpretation and achieve forgery localization guided by detailed textual descriptions. Extensive experiments demonstrate that FakeShield effectively detects and localizes various tampering techniques, offering an explainable and superior solution compared to previous IFDL methods.

![image](https://github.com/user-attachments/assets/9146a162-31aa-4db6-beb2-97a6207ec0c6)

