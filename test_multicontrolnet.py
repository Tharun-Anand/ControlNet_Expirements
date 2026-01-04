#!/usr/bin/env python3
"""
Minimal inference testing script for Multi-ControlNet SemanticControl
"""

import os
import sys
import torch
from pytorch_lightning import seed_everything

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Assumes the previous EvalModel code is saved in test/inference.py
from test.inference_multicontrol import EvalModel
from test.types import ModelType

def test_multicontrolnet(prompt="a cyberpunk detective in a rainy city", 
                    references=[
                        "/home/saketh/repos/SemanticControl/assets/test/pose_ref.jpg",
                        "/home/saketh/repos/SemanticControl/assets/test/depth_ref.jpg",
                        "/home/saketh/repos/SemanticControl/assets/test/canny_ref.jpg"
                    ],
                    ref_subj="a person", 
                    prmpt_subj="a detective", 
                    seed=1, 
                    alpha_mask=[1.0], 
                    control_types=["pose", "depth", "canny"], 
                    output_dir="./test_outputs_exp1/multicontrol_res",
                    indices=None):
    """
    Test Multi-ControlNet inference.
    
    :param references: List of file paths to reference images.
    :param control_types: List of control types corresponding to the references.
    """
    print(f"Testing Multi-ControlNet inference with controls: {control_types}...")
    
    # 1. Initialize model with LIST of controls
    # Matches EvalModel.__init__(self, controls: List[str])
    eval_model = EvalModel(controls=control_types)
    eval_model.set_output_dir(output_dir)
    
    # 2. Get inference function
    inference_func = eval_model.get_inference_func(ModelType.ControlNet)
    
    # 3. Run inference
    # Matches EvalModel.inference_ControlNet(..., references: List[str], ...)
    try:
        output = inference_func(
            prompt=prompt,
            references=references,  # PASSING LIST HERE
            ref_subj=ref_subj,
            prmpt_subj=prmpt_subj,
            seed=seed,
            alpha_mask=alpha_mask,
            indices=indices
        )
        
        # 4. Post-process (Save Grid)
        if output:
            eval_model.postprocess(output, save_attn=True)
            print("✅ Multi-ControlNet test successful!")
            return True
        else:
            print("⚠️ Output was None (likely already generated).")
            return True

    except Exception as e:
        print(f"❌ Multi-ControlNet test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_requirements(image_paths):
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check if all test images exist
    missing_images = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            missing_images.append(img_path)
    
    if missing_images:
        print(f"❌ Missing test images: {missing_images}")
        return False
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, will use CPU (slower)")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    print("✅ Requirements check passed")
    return True

def main():
    """Main test function configuration"""
    print("🚀 Starting SemanticControl Multi-ControlNet Tests")
    print("=" * 50)

    # --- CONFIGURATION ---
    # Define the controls you want to use together
    # Options: "sketch", "pose", "depth", "canny"
    my_controls = ["canny", "pose"]

    # Define the reference images corresponding to those controls
    # Ensure these paths are correct on your system!
    base_asset_path = "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test"
    my_references = [
       "/scratch/work/muralit1/Controlnet/SemanticControl/assets/bg/background1.png",
       "/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a man is holding a sword.jpg"
    ]
    
    prompt = "a man in the woods"
    ref_subj = "a woman"
    prmpt_subj = "a dog plushie"
    seed = 83
    output_dir = "./test_outputs_exp1/multicontrol_res"

    # --- EXECUTION ---

    # Check requirements
    if not check_requirements(my_references):
        print("❌ Requirements not met. Exiting.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(seed)
    
    tests = [
        ("MultiControlNet", lambda: test_multicontrolnet(
            prompt=prompt, 
            references=my_references, 
            ref_subj=ref_subj, 
            prmpt_subj=prmpt_subj, 
            seed=seed, 
            control_types=my_controls, 
            output_dir=output_dir,
            indices=None
        )),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n▶️ Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()