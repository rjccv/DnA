<div align="center">
<h5>

<h2><a href="https://arxiv.org/abs/2510.13808" style="color:#9C276A">
VisCoP: Visual Probing for Domain Adapatation of Vision Language Models</a></h2>

[![arXiv](https://img.shields.io/badge/arXiv-VisCoP%20Paper-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2510.13808)
[![HuggingFace ](https://img.shields.io/badge/🤗%20HuggingFace-Training%20Data-FFD21F?style=flat)](https://huggingface.co/datasets/dreilly/VisCoP_data)

</h5>

<p align="center">
<img src="assets/teaser.jpg" width="1000px" >
</p>

</div>

<p align="center">
  <video src="https://github.com/user-attachments/assets/3478722b-e98b-40e8-a82a-b15a9770c542" autoplay loop muted></video>
</p>


## About This Repository
This directory contains a modified version of the original VisCoP codebase ([paper](https://arxiv.org/abs/2510.13808), [official repository](https://github.com/dominickrei/VisCoP)).

Main modification in this fork:
- Replaced the original probe cross-attention with a denoising attention mechanism for our DnA-based experiments.



## ⚙️ Installation
1. Create a conda environment
```shell
conda create --name=viscop python=3.10
conda activate viscop
```

2. Clone VisCoP and install the required Python packages (we use `torch 2.4.0 + cuda 12.4` in our experiments)
```
git clone https://github.com/dominickrei/VisCoP.git
cd VisCoP
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
```

## 🏋️ Training VisCoP
### 🎥 Preparing Training Data for Egocentric Viewpoint and Depth Modality
We provide the instruction pairs as well as videos for training through [HuggingFace](https://huggingface.co/datasets/dreilly/VisCoP_data). After downloading the data, update the following variables in `scripts/train/ego_depth_video/train_viscop_dna.slurm`:
* `DATA_DIR`: Update with the path to either egocentric or depth videos
* `TRAINING_JSON`: Update with the path to either egocentric or depth instructions


**Coming soon!**

### 🔥 Update Training Script and Launch Training
In `scripts/train/ego_depth_video/train_viscop_dna.sh`, update the following arguments to match your system settings and paths:
* `INIT_MODEL`: This is the path to weights of the base VLM (VideoLLaMA3). Please use the following command to download and save the weights `python scripts/save_basevlm_for_finetuning.py --save-path-for-local-basevlm /path/to/save/base_vlm`
* `DATA_DIR`: The path to your data directory containing the egocentric, depth, or robot control data
* `TRAINING_JSON`: The path to a json file containing the egocentric, depth, or robot control instructions
* (Optional) `NUM_VISUAL_PROBES`: The number of Visual Probes to use in VisCoP
* (Optional) `INTERACTION_MODULE_POS`: The positions of the interaction modules. Acceptable values are `all` or a comma-separated list of integers (denoting zero-indexed layer indices of the vision encoder)


(**Training with SLURM**) After updating the training scripts, update the arguments in `train_viscop_dna.sh` and submit the job:
```shell
cd scripts/train/ego_depth_video
sbatch train_viscop_dna.sh
```

## ❄️ Evaluating VisCoP
### 💾 Preparing Source and Target Domain Data
| Target domain | Datasets |
|-----------|-----------|
| Egocentric Viewpoint | [Ego-in-Exo PerceptionMCQ](https://huggingface.co/datasets/dreilly/Ego-in-Exo-PerceptionMCQ), [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema) |

<details>
<summary>Click to view our evaluation directory structure</summary>
  
    /path/to/vlm_eval_bench/
    ├── egoperceptionmcq
    │   ├── all_category_qas.json
    │   ├── keystep_segments
    │   └── depth_videos
    ├── egoschema
        ├── GENERATION
        ├── MC
        ├── MC_PPL
        ├── questions.json
        ├── Subset
        ├── subset_answers.json
        └── videos


</details>

### 🏃🎥 Run the video understanding evaluations
After downloading the data, update the following variables in `scripts/eval/eval_video.sh`:
* `DATA_ROOT`: Update with the path to your evaluation directory, ensure it follows the same structure as shown above

After updating the evaluation script, run the following command:
```shell
cd scripts/train/ego_depth_video
sbatch eval_dna.sh
```

**NOTE:** Evaluations on Ego-in-Exo PerceptionMCQ and ADL-X require Llama 3.1. You will need to install [Ollama](https://ollama.com/download) and download the Llama 3.1 model by running the command `ollama run llama3.1` prior to running the evaluations.
* If you are using an HPC environment and can not install Ollama, you will need to run an Ollama server locally
  * To do this, download the Ollama server that matches your system architecture from their [releases page](https://github.com/ollama/ollama/releases). Then update and uncomment lines `59-62` in `scripts/eval/eval_video.sh`
 


## 🙏 Acknowledgements
We thank the researchers behind the following codebases and model releases for their great open source work which VisCoP builds upon! [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3), [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [SigLIP](https://arxiv.org/abs/2303.15343), and [Qwen2.5](https://arxiv.org/abs/2412.15115).






