<div align="center">
  <h1>MedShapeNet: A Large-scale Dataset of 3D Medical Shapes</h1>
  
  <p>
    <a href="https://arxiv.org/pdf/2308.16139.pdf"><img src="https://img.shields.io/badge/Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="Paper" /></a>
    <a href="http://jianningli.me/pdfs/MedShapeNet%20JHU%20Presentation%20Jianning.pdf"><img src="https://img.shields.io/badge/Presentation-FF8C00?style=flat-square&logo=microsoft-powerpoint&logoColor=white" alt="Presentation" /></a>
    <a href="https://medshapenet-ikim.streamlit.app/"><img src="https://img.shields.io/badge/Website-000000?style=flat-square&logo=google-chrome&logoColor=white" alt="Website" /></a>
    <a href="https://medshapenet.ikim.nrw/uploads/MedShapeNetDataset.txt"><img src="https://img.shields.io/badge/Download-0078D4?style=flat-square&logo=microsoft-onedrive&logoColor=white" alt="Download" /></a>
  </p>

  <p>
    <a href="https://github.com/Jianningli/medshapenet-feedback/tree/main/pip_install_MedShapeNetCore"><code>pip install MedShapeNetCore</code></a> • 
    <a href="https://github.com/Jianningli/medshapenet-feedback/blob/main/pip_install_MedShapeNetCore/examples/MedShapeNetShowCase.ipynb">MedShapeNetCore Showcase</a>
  </p>
  
  <img src="https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/github.png" alt="gallery" width="80%">
</div>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Contribution Guidelines](#-contribution-guidelines)
  - [Report Issues](#report-issues)
  - [Contribute Shapes](#contribute-shapes)
  - [Showcase Your Research](#showcase-your-research)
  - [Suggest Improvement](#suggest-improvement)
- [Installation](#-installation)
- [References](#-references)
- [Contributors](#-contributors)
- [Related Publications](#-related-publications)
- [Contact](#-contact)

---

## 📖 Overview

**MEDSHAPENET FEEDBACK** is a platform for researchers to contribute shapes, provide feedback (e.g., report corrupted shapes for removal, suggest improvements), and showcase their own research and applications utilizing MedShapeNet. 

It is an important means of communication for MedShapeNet developers, users, and contributors to continuously refine the database and promote the translation of shape-related methods from computer vision to medical applications. Shape contributors have the chance of being listed as collaborators of the MedShapeNet project upon request. 

For more details about the project, please check out our [paper](https://arxiv.org/pdf/2308.16139.pdf) and [website](https://medshapenet-ikim.streamlit.app/).

---

## 🤝 Contribution Guidelines 

By [filing a pull request](https://github.com/Jianningli/medshapenet-feedback/pulls) or [opening an issue](https://github.com/Jianningli/medshapenet-feedback/issues) in this repository, you can:

- 🐛 **Report Issues**: Report to us corrupted/incorrect/unusable shapes you found, or request removal of certain shapes if you are the owner of the original datasets. [[Issue](https://github.com/Jianningli/medshapenet-feedback/issues)]
- ➕ **Contribute Shapes**: Contribute medical shapes extracted from your own datasets. [[Issue](https://github.com/Jianningli/medshapenet-feedback/issues)]
- 💡 **Showcase Research/Applications**: Describe your research/project that utilizes MedShapeNet by creating a new folder in this repository (following [this example](https://github.com/Jianningli/medshapenet-feedback/tree/main/anatomy-completor)). [[Pull Request](https://github.com/Jianningli/medshapenet-feedback/pulls)]
- 🚀 **Suggest Improvement**: Tell us the desired functions you want in the [MedShapeNet web interface](https://medshapenet-ikim.streamlit.app/). [[Issue](https://github.com/Jianningli/medshapenet-feedback/issues)]

### Templates for Issues and Pull Requests:

<details>
<summary><b>Report Issues</b></summary>

- **Search query of the shape(s):**
- **Description of the problem:**
- **(Optional) Screenshot of the problematic shape(s):**
</details>

<details>
<summary><b>Contribute Shapes</b></summary>

- **Link to dataset(s):**
- **Description of the dataset(s):** Publications, technical reports, etc.
- **Contributor information:** Name, affiliation, homepage
- **Other comments:**
</details>

<details>
<summary><b>Showcase Your Research</b></summary>

Formatting is flexible. You can find existing examples [here](https://github.com/Jianningli/medshapenet-feedback/tree/main/anatomy-completor) and [here](https://github.com/Jianningli/medshapenet-feedback/tree/main/forensic-facial-reconstruction).
</details>

<details>
<summary><b>Suggest Improvement</b></summary>

Provide details on what you would like to see improved.
</details>

---

## ⚙️ Installation

```bash
pip install MedShapeNetCore
```

> **Note**: For detailed usage, refer to this [Google Colab](https://colab.research.google.com/github/Jianningli/medshapenet-feedback/blob/main/pip_install_MedShapeNetCore/getting_started.ipynb).

---

## 📚 References 

If you use MedShapeNet in your research, please cite MedShapeNet as:

```bibtex
@article{li2023medshapenet,
  title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
  author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
  journal={arXiv preprint arXiv:2308.16139},
  year={2023}
}
```

---

## 👥 Contributors 
Refer to our [MedShapeNet Paper](https://arxiv.org/pdf/2308.16139.pdf) for a full list of contributors to the project.

## 📄 Related Publications
Refer to the [publication page](https://proj-page.github.io/medshapenet_publications.html).

## 📬 Contact 
Contact **Jianning Li** ([jianningli.me@gmail.com](mailto:jianningli.me@gmail.com)) for any questions related to MedShapeNet.
