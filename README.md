# FLUX.1-dev LoRA Explorer Cog Model

This is an implementation of [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) LoRA inference as a [Cog](https://github.com/replicate/cog) model.


## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [Cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="a Witch, Linear red light" -i hf_lora="https://huggingface.co/Shakker-Labs/SD3.5-LoRA-Linear-Red-Light/resolve/main/SD35-lora-Linear-Red-Light.safetensors"

![output](output.0.png)


## License

The code in this repository is licensed under the [Apache-2.0 License](LICENSE).

Stable Diffusion 3.5-Large falls under the [`STABILITY AI COMMUNITY` License](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md).

`Stable Diffusion 3.5-Large` fine-tuned weights and their outputs are non-commercial by default, but can be used commercially when running on [Replicate](https://replicate.com).
