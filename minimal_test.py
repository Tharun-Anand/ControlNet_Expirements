#!/usr/bin/env python3
"""
Minimal inference testing script for SemanticControl
"""

import os
import sys
import torch
from PIL import Image
from pytorch_lightning import seed_everything

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test.inference import EvalModel
from test.types import ModelType

def test_controlnet(prompt="a dog plushie riding a bike", 
                   reference="/home/saketh/repos/SemanticControl/assets/test/a woman is riding a bike.jpg",
                   ref_subj="a woman", prmpt_subj="a dog plushie", 
                   seed=1, alpha_mask=[1.0], control_type="depth", output_dir= "./test_outputs_exp1/no_up1block_res",indices = None):
    """Test basic ControlNet inference"""
    print("Testing ControlNet inference...")
    
    # Initialize model
    eval_model = EvalModel(control=control_type)  # or "canny", "pose"
    eval_model.set_output_dir(output_dir)
    
    # Get inference function
    inference_func = eval_model.get_inference_func(ModelType.ControlNet)
    
    # Run inference
    # try:
    output = inference_func(
        prompt=prompt,
        reference=reference,
        ref_subj=ref_subj,
        prmpt_subj=prmpt_subj,
        seed=seed,
        alpha_mask=alpha_mask,
        indices=indices
    )
    
    # if output:
    eval_model.postprocess(output, save_attn = True)
    print("✅ ControlNet test successful!")
    return True
    # else:
    #     print("❌ ControlNet test failed - no output generated")
    #     return False
            
    # except Exception as e:
    #     print(f"❌ ControlNet test failed with error: {e}")
    #     return False

def test_semantic_control(prompt="a dog plushie riding a bike", 
                         reference="/home/saketh/repos/SemanticControl/assets/test/a woman is riding a bike.jpg",
                         ref_subj="a woman", prmpt_subj="a dog plushie", 
                         seed=1, mask_prompt="A woman riding a bike", 
                         focus_tokens="riding a bike", save_attn=False, 
                         use_attn_bias=True, control_type="depth"):
    """Test SemanticControl inference"""
    print("Testing SemanticControl inference...")
    
    # Initialize model
    eval_model = EvalModel(control=control_type)  # or "canny", "pose"
    eval_model.set_output_dir("./test_outputs")
    
    # Get inference function
    inference_func = eval_model.get_inference_func(ModelType.SemanticControl)
    
    # Run inference
    try:
        output = inference_func(
            prompt=prompt,
            reference=reference,
            ref_subj=ref_subj,
            prmpt_subj=prmpt_subj,
            seed=seed,
            mask_prompt=mask_prompt,
            focus_tokens=focus_tokens,
            save_attn=save_attn,
            use_attn_bias=use_attn_bias
        )
        
        if output:
            eval_model.postprocess(output, save_attn = True)
            print("✅ SemanticControl test successful!")
            return True
        else:
            print("❌ SemanticControl test failed - no output generated")
            return False
            
    except Exception as e:
        print(f"❌ SemanticControl test failed with error: {e}")
        return False

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check if test images exist
    test_images = [
        "./assets/test/a man is holding the guitar.jpg",
        "./assets/test/a man is holding a sword.jpg", 
        "./assets/test/a woman is riding a bike.jpg"
    ]
    
    missing_images = []
    for img_path in test_images:
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

def main(prompt="a cat holding a sword", 
         reference="/scratch/work/muralit1/Controlnet/SemanticControl/assets/test/a man is holding a sword.jpg",
         ref_subj="a man", prmpt_subj="a cat", 
         seed=83, alpha_mask=[1.0], mask_prompt="A man holding a sword", 
         focus_tokens="holding a sword", save_attn=False, 
         use_attn_bias=True, control_type="depth",output_dir = "./test_outputs_exp1/no_up1block_res", indices = None):
    """Main test function with all prompt-related arguments"""
    print("�� Starting SemanticControl Minimal Tests")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements not met. Exiting.")
        return
    
    # Create output directory
    # output_dir = "./test_outputs_exp1/no_up1block_res"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    tests = [
        ("ControlNet", lambda: test_controlnet(
            prompt=prompt, reference=reference, ref_subj=ref_subj, 
            prmpt_subj=prmpt_subj, seed=seed, alpha_mask=alpha_mask, 
            control_type=control_type, output_dir=output_dir,indices= indices
        )),
        # ("SemanticControl", lambda: test_semantic_control(
        #     prompt=prompt, reference=reference, ref_subj=ref_subj, 
        #     prmpt_subj=prmpt_subj, seed=seed, mask_prompt=mask_prompt, 
        #     focus_tokens=focus_tokens, save_attn=save_attn, 
        #     use_attn_bias=use_attn_bias, control_type=control_type
        # ))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n�� Running {test_name} test...")
        # try:
        result = test_func()
        results.append((test_name, result))
        # except Exception as e:
        #     print(f"❌ {test_name} test crashed: {e}")
        #     results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("�� TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("�� All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()
