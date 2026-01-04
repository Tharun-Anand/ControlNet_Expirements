import os
from typing import List, TypedDict, Optional

import torch
# Added HEDdetector/Midas for Sketch/Depth robustness if needed
from controlnet_aux import CannyDetector, OpenposeDetector, ZoeDetector, HEDdetector 
from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline
from PIL import Image
from pytorch_lightning import seed_everything

from lib import (assert_path, image_grid, init_store_attn_map,
                 save_alpha_masks, save_attention_maps)
from src import SemanticControlPipeline, register_unet

from .types import ModelType

class GenerateParam(TypedDict):
    seed: int
    prompt: str
    ignore_special_tkns: bool

class EvalModel():
    base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    
    # Stores the list of active control names, e.g. ["pose", "depth"]
    controls: List[str] = [] 
    generate_param: GenerateParam = {}

    def _control_setup(self, controls: List[str]):
        """
        Sets up a LIST of ControlNet models and Preprocessors based on the input list.
        """
        self.controlnet_models = []
        self.preprocessors = []
        
        print(f"Setting up Multi-ControlNet for: {controls}")

        for control in controls:
            if control == "depth":
                self.controlnet_models.append(
                    ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
                )
                self.preprocessors.append(ZoeDetector.from_pretrained("lllyasviel/Annotators"))
                
            elif control == "canny":
                self.controlnet_models.append(
                    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
                )
                self.preprocessors.append(CannyDetector())
                
            elif control == "pose":
                self.controlnet_models.append(
                    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
                )
                self.preprocessors.append(OpenposeDetector.from_pretrained("lllyasviel/Annotators"))
                
            elif control == "sketch":
                # Assuming 'scribble' model for sketch
                self.controlnet_models.append(
                    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16)
                )
                self.preprocessors.append(HEDdetector.from_pretrained("lllyasviel/Annotators"))
            
            else:
                raise ValueError(f"Unknown control type: {control}")

    def __init__(self, controls: List[str]):
        """
        :param controls: List of control types, e.g. ["pose", "canny"]
        """
        self.controls = controls
        self._control_setup(controls)

        vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)
        
        # Pass the LIST of models to the pipeline
        pipe = SemanticControlPipeline.from_pretrained(
            pretrained_model_name_or_path=self.base_model_path,
            controlnet=self.controlnet_models, 
            vae=vae,
            torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        self.pipe = pipe
        
    def _prepare_control(self, references: List[str]) -> List[Image.Image]:
        """
        Process a LIST of reference image paths using the corresponding preprocessors.
        """
        if len(references) != len(self.controls):
            raise ValueError(f"Number of reference images ({len(references)}) does not match number of controls ({len(self.controls)})")

        self.reference_imgs = [] # Store original loaded images
        control_maps = []        # Store processed maps
        
        for idx, ref_path in enumerate(references):
            print(f"Preprocessing image {idx+1}/{len(references)} for {self.controls[idx]}...")
            
            # Load original
            orig_image = Image.open(ref_path).convert("RGB")
            self.reference_imgs.append(orig_image)
            
            # Apply corresponding preprocessor
            processed_map = self.preprocessors[idx](orig_image)
            # 3. FIX: Resize to 512x512
            processed_map = processed_map.resize((512, 512), Image.LANCZOS)

            control_maps.append(processed_map)

        # Store for postprocess usage
        self.control_maps = control_maps
        return control_maps

    def _record_generate_params(self, seed: int,  prompt: str, ignore_special_tkns: bool):
        self.generate_param["seed"] = seed
        self.generate_param["prompt"] = prompt
        self.generate_param["ignore_special_tkns"] = ignore_special_tkns

    def get_inference_func(self, modelType: ModelType):
        self.modelType = modelType
        if modelType == ModelType.ControlNet:
            return self.inference_ControlNet
        elif modelType == ModelType.SemanticControl:
            return self.inference_SemanticControl
        return None

    def set_output_dir(self, output_dir: str):
        assert_path(output_dir)
        self.output_dir = output_dir

    @staticmethod
    def _filepath2name(filepath: str):
        no_extension = filepath.split(".")[0]
        name = " ".join(no_extension.split("/")[-2:])
        return name

    def _is_already_generated(self, ref_subj, prmpt_subj, prompt, seed, alpha_mask=[1.0], prefix=""):
        # Create folder name based on combined controls
        controls_name = "_".join(self.controls)
        save_dir = f"{self.output_dir}/{ref_subj}/{prmpt_subj}/{prompt}/{self.modelType.name}/{controls_name}"
        
        filename = f"{save_dir}/{prefix}seed {seed}.png"
        if self.modelType.name == "ControlNet":
             filename = f"{save_dir}/alpha {alpha_mask} - seed {seed}.png"

        self.save_dir = save_dir
        self.filename = filename

        return os.path.exists(filename)

    def inference_ControlNet(self, prompt: str, references: List[str], ref_subj: str, prmpt_subj: str, seed: int, alpha_mask: List[float] = [1.0], indices: Optional[List[int]] = None, **kwargs):
        """
        :param references: List of paths to input images corresponding to self.controls order
        """
        # Check cache
        if self._is_already_generated(ref_subj, prmpt_subj, prompt, seed, alpha_mask):
            print(f"{self.filename} is already generated. Skipping.")
            return None

        # 1. Prepare List of Control Maps
        control_maps = self._prepare_control(references)
        
        pipe_options = {
            "ignore_special_tkns": True
        }
        self.pipe.options = pipe_options
        
        init_store_attn_map(self.pipe)
        register_unet(self.pipe, mask_options={
            "alpha_mask": alpha_mask,
            "fixed": True
        })

        self._record_generate_params(seed=seed, prompt=prompt, ignore_special_tkns=True)

        seed_everything(seed)
        print('indices:', indices)

        # 2. Define Scales (Default 1.0 for all)
        scales = [1.0] * len(control_maps)

        # 3. Inference
        output = self.pipe(
            prompt=prompt,
            image=control_maps,
            height=512, # Explicitly ensure generation size matches control maps
            width=512,
            indices=indices,
            controlnet_conditioning_scale=scales
        ).images[0]

        return output

    def postprocess(self, image, save_attn: bool = False):
        if image is None:
            return 
        
        assert_path(self.save_dir)
        image.resize((512, 512)).save(self.filename)

        grid_imgs = []
        if hasattr(self, 'reference_imgs') and hasattr(self, 'control_maps'):
            for idx, (ref, map_img) in enumerate(zip(self.reference_imgs, self.control_maps)):
                control_name = self.controls[idx]
                map_img.resize((512, 512)).save(f"{self.save_dir}/{control_name}_condition.png")
                grid_imgs.append(ref.resize((512, 512)))
                grid_imgs.append(map_img.resize((512, 512)))

        grid_imgs.append(image.resize((512, 512)))
        comparison = image_grid(grid_imgs, 1, len(grid_imgs))
        comparison.save(f"{self.save_dir}/multicontrol_result_seed_{self.generate_param['seed']}.png")

        print(f"Saved results for {self.filename}")

    # SemanticControl Placeholder
    def inference_SemanticControl(self, prompt: str, reference: str, ref_subj: str, prmpt_subj: str, seed: int, mask_prompt, focus_tokens, save_attn: bool = False, **kwargs):
        pass