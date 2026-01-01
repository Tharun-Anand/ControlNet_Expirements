#!/usr/bin/env python3
"""
Ablation study runner:
Each index combination is treated as ONE experiment.
All prompts are run inside that experiment.
"""

import os
import sys
from pytorch_lightning import seed_everything

# =====================================
# Project setup
# =====================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from minimal_test import main  # updated main()


# =====================================
# Root directory for ablation logs
# =====================================
ABLATION_ROOT = "ablation_study"


# =====================================
# Index experiments (ABLATIONS)
# =====================================
INDEX_EXPERIMENTS = [
    [0],
    [1],
    [2],
    [3],
    # add more here
]


# =====================================
# Prompt configurations
# =====================================
PROMPTS = {
    "girl_with_purse": {
        "prompt": "a girl holding a purse in anime style",
        "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/hulk.png",
        "ref_subj": "a hulk",
        "prmpt_subj": "a girl",
        "seed": 30,
        "alpha_mask": [1.0],
        "mask_prompt": "A hulk holding a purse",
        "focus_tokens": "holding a purse",
        "save_attn": True,
        "use_attn_bias": True,
        "control_type": "depth",
    },

    "cat_holding_guitar": {
        "prompt": "a cat holding a guitar",
        "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a man is holding the guitar.jpg",
        "ref_subj": "a man",
        "prmpt_subj": "a cat",
        "seed": 42,
        "alpha_mask": [1.0],
        "mask_prompt": "A man holding a guitar",
        "focus_tokens": "holding a guitar",
        "save_attn": True,
        "use_attn_bias": True,
        "control_type": "pose",
    },
    "boy_with_sword": {
        "prompt": "a cat holding a sword",
        "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a man is holding a sword.jpg",
        "ref_subj": "a man",
        "prmpt_subj": "a cat",
        "seed": 87,
        "alpha_mask": [1.0],
        "mask_prompt": "A man holding a sword",
        "focus_tokens": "holding a sword",
        "save_attn": True,
        "use_attn_bias": True,
        "control_type": "depth",
    },
}


# =====================================
# Helpers
# =====================================
def index_exp_name(indices):
    """[1,5,2] -> index_1_5_2"""
    return "index_" + "_".join(map(str, indices))


# =====================================
# Runner
# =====================================
def run_ablation_study():
    print("🚀 Starting ablation study")
    print("=" * 80)

    for indices in INDEX_EXPERIMENTS:
        exp_name = index_exp_name(indices)
        exp_path = os.path.join(ABLATION_ROOT, exp_name)

        print(f"\n🧪 Index Experiment: {exp_name}")
        print(f"📁 Log path: {exp_path}")
        print("-" * 80)

        # ---------------------------------
        # Create experiment directory
        # ---------------------------------
        os.makedirs(exp_path, exist_ok=True)

        # Optional: fix seed per index experiment
        base_seed = 123 + sum(indices)
        seed_everything(base_seed)

        # ---------------------------------
        # Run all prompts
        # ---------------------------------
        for prompt_name, cfg in PROMPTS.items():
            print(f"   ▶ Prompt: {prompt_name}")

            # Optional: subfolder per prompt
            prompt_path = os.path.join(exp_path, prompt_name)
            os.makedirs(prompt_path, exist_ok=True)

            try:
                main(
                    prompt=cfg["prompt"],
                    reference=cfg["reference"],
                    ref_subj=cfg["ref_subj"],
                    prmpt_subj=cfg["prmpt_subj"],
                    seed=cfg["seed"],
                    alpha_mask=cfg["alpha_mask"],
                    mask_prompt=cfg["mask_prompt"],
                    focus_tokens=cfg["focus_tokens"],
                    save_attn=cfg["save_attn"],
                    use_attn_bias=cfg["use_attn_bias"],
                    control_type=cfg["control_type"],
                    indices=indices,           # 🔑 ablation variable
                    output_dir=prompt_path,          # 🔑 log directory
                )
            except Exception as e:
                print(f"   ❌ Failed prompt: {prompt_name}")
                print(f"      Error: {e}")


# =====================================
# Entry point
# =====================================
if __name__ == "__main__":
    run_ablation_study()
