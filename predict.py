# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import shutil
import subprocess
import numpy as np
from typing import List
from PIL import ImageOps
from diffusers.utils import load_image
from weights import WeightsDownloadCache
from transformers import CLIPImageProcessor
from diffusers.image_processor import VaeImageProcessor
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline
)
from PIL import ImageOps
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

SD3_MODEL_CACHE = "./stable-diffusion-3.5-large"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SD3_URL = "https://weights.replicate.delivery/default/stabilityai/stable-diffusion-3.5-large/model.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1360, 768),
    "21:9": (1568, 672),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "4:5": (912, 1136),
    "5:4": (1136, 912),
    "3:4": (880, 1184),
    "4:3": (1184, 880),
    "9:16": (768, 1360),
    "9:21": (672, 1568),
}

def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        self.weights_cache = WeightsDownloadCache()
        self.last_loaded_lora = None

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SD3_MODEL_CACHE):
            download_weights(SD3_URL, SD3_MODEL_CACHE)

        print("Loading sd3 txt2img pipeline...")
        self.txt2img_pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.txt2img_pipe.to("cuda")

        print("Loading sd3 img2img pipeline...")
        self.img2img_pipe = StableDiffusion3Img2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            text_encoder_3=self.txt2img_pipe.text_encoder_3,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            tokenizer_3=self.txt2img_pipe.tokenizer_3,
            transformer=self.txt2img_pipe.transformer,
            scheduler=self.txt2img_pipe.scheduler,
        )

        # fix for img2img
        # self.img2img_pipe.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=self.img2img_pipe.vae.config.latent_channels)
        self.img2img_pipe.to("cuda")
        print("setup took: ", time.time() - start)


    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        tmp_img = load_image("/tmp/image.png").convert("RGB")
        return ImageOps.contain(tmp_img, (1024, 1024))

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept
    
    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]
    
    def load_single_lora(self, lora_url: str, pipe):
        # Clean up the previous lora and load the new one
        pipe.unload_lora_weights()
        try:
            lora_path = self.weights_cache.ensure(lora_url)
            pipe.load_lora_weights(lora_path)
            self.last_loaded_lora = lora_url
        except Exception as e:
            raise Exception(f"Invalid lora, must be either a: Replicate path, Huggingface URL, CivitAI URL, or a URL to a .safetensors file: {lora_url}")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="",
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=list(ASPECT_RATIOS.keys()),
            default="1:1"
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=20, default=4.5
        ),
        image: Path = Input(
            description="Input image for img2img mode",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.7,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommended range is 28-50",
            ge=1,
            le=50,
            default=28,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        hf_lora: str = Input(
            description="Full URL to Huggingface LoRA, CivitAI URL, or URL to a .safetensors file",
            default=None,
        ),
        lora_scale: float = Input(
            description="Scale for the LoRA weights",
            ge=0,le=1, default=0.8,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        num_outputs = 1

        sd3_kwargs = {}
        print(f"Prompt: {prompt}")
        if image:
            print("img2img mode")
            sd3_kwargs["image"] = self.load_image(image)
            sd3_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sd3_kwargs["width"] = width
            sd3_kwargs["height"] = height
            pipe = self.txt2img_pipe

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if hf_lora:
            sd3_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
            t1 = time.time()
             # check if extra_lora is new
            if self.last_loaded_lora != hf_lora:
                self.load_single_lora(hf_lora, pipe)
            t2 = time.time()
            print(f"Loading LoRA took: {t2 - t1:.2f} seconds")
        else:
            sd3_kwargs["joint_attention_kwargs"] = None
            pipe.unload_lora_weights()
            self.last_loaded_lora = None

        output = pipe(**common_args, **sd3_kwargs)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
