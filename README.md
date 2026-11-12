# Mamba from scratch, for speech people

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bartkw12/mamba-speech-demo/blob/main/mamba_speech_demo.ipynb)

An interactive notebook that builds the core of **Mamba** (Gu & Dao, 2023), a selective state space model, in plain PyTorch, and uses it on speech.
Made for the Mamba model presentation in CAS 784, *Speech Recognition in Humans and Machines* (McMaster University, Nov 2026), by Bart Kowal and Xiao Shao.

No GPU, CUDA kernels or installs needed: click **Open in Colab** and run the cells top to bottom.

## What's inside

| Part | What you'll see |
| --- | --- |
| 1 | A selective scan in about 20 lines, with sliders for the step size Δ |
| 2 | Attention vs scan: run time as the input gets longer |
| 3 | Why selection matters: a "remember the name" task, selective vs plain SSM |
| 4 | Tiny Mamba vs tiny Transformer on the Free Spoken Digit Dataset |
| 5 | Look inside: what the model writes to memory, plus its mistakes |
| 6 | Optional: run a real pretrained Mamba language model |
| 7 | Model autopsy card |

## Sources

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Zhang, X., et al. (2025). *Mamba in Speech: Towards an Alternative to Self-Attention.* IEEE/ACM TASLP 33, 1933–1948. [doi:10.1109/TASLPRO.2025.3566210](https://doi.org/10.1109/TASLPRO.2025.3566210)
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) (CC BY-SA 4.0), downloaded when the notebook runs.

The Mamba block here is a simplified teaching version, not the official implementation ([state-spaces/mamba](https://github.com/state-spaces/mamba)).
