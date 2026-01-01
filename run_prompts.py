#!/usr/bin/env python3
"""
Batch runner for SemanticControl experiments
"""
import os
import sys
from pytorch_lightning import seed_everything

# Add path so Python can find your test script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# 🔽 IMPORT YOUR EXISTING SCRIPT
from minimal_test import main   # <-- change filename if needed


# ================================
# Experiment configurations
# ================================
EXPERIMENTS = {
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
    "boy_with_sword": {
        "prompt": "a girl holding a purse in anime style",
        "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/hulk.png",
        "ref_subj": "a hulk",
        "prmpt_subj": "a girl",
        "seed": 29,
        "alpha_mask": [1.0],
        "mask_prompt": "A hulk holding a purse",
        "focus_tokens": "holding a purse",
        "save_attn": True,
        "use_attn_bias": True,
        "control_type": "depth",
    },
    # "boy_with_sword": {
    #     "prompt": "a boy holding a wooden hammer",
    #     "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a man is holding a sword.jpg",
    #     "ref_subj": "a man",
    #     "prmpt_subj": "a boy",
    #     "seed": 28,
    #     "alpha_mask": [1.0],
    #     "mask_prompt": "A man holding a wooden hammer",
    #     "focus_tokens": "holding a wooden hammer",
    #     "save_attn": True,
    #     "use_attn_bias": True,
    #     "control_type": "depth",
    # }

    # "cat_riding_bike": {
    #     "prompt": "a cat riding a bike",
    #     "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a woman is riding a bike.jpg",
    #     "ref_subj": "a woman",
    #     "prmpt_subj": "a cat",
    #     "seed": 1,
    #     "alpha_mask": [1.0],
    #     "mask_prompt": "A woman riding a bike",
    #     "focus_tokens": "riding a bike",
    #     "save_attn": True,
    #     "use_attn_bias": True,
    #     "control_type": "depth",
    # },

    "boy_holding_guitar": {
        "prompt": "a cat holding a guitar",
        "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a man is holding the guitar.jpg",
        "ref_subj": "a man",
        "prmpt_subj": "a cat",
        "seed": 41,
        "alpha_mask": [1.0],
        "mask_prompt": "A man holding a guitar",
        "focus_tokens": "holding a guitar",
        "save_attn": True,
        "use_attn_bias": True,
        "control_type": "pose",
    },
    # "cat_with_sword": {
    #     "prompt": "a cat is holding the sword",
    #     "reference": "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a man is holding a sword.jpg",
    #     "ref_subj": "a man",
    #     "prmpt_subj": "a cat",
    #     "seed": 29,
    #     "alpha_mask": [1.0],
    #     "mask_prompt": "A man holding a sword",
    #     "focus_tokens": "holding a sword",
    #     "save_attn": True,
    #     "use_attn_bias": True,
    #     "control_type": "depth",
    # }
}

# ================================
# Batch execution
# ================================
def run_all_experiments():
    print("🚀 Running batch SemanticControl experiments")
    print("=" * 70)

    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n🧪 Experiment: {exp_name}")
        print("-" * 70)

        # Optional: make per-experiment output folder
        os.environ["EXP_NAME"] = exp_name

        seed_everything(cfg["seed"])

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
            )
        except Exception as e:
            print(f"❌ Experiment {exp_name} failed: {e}")


if __name__ == "__main__":
    run_all_experiments()