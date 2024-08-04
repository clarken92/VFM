# Link: https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from os.path import join, exists

from PIL import Image
from matplotlib import pyplot as plt

from tqdm.auto import tqdm

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, \
    UNet2DConditionModel, PNDMScheduler
from diffusers import \
    DDIMScheduler, UniPCMultistepScheduler, \
    DPMSolverSinglestepScheduler, DEISMultistepScheduler, \
    PNDMScheduler, IPNDMScheduler, \
    DPMSolverMultistepScheduler

repo_id = "CompVis/stable-diffusion-v1-4"

# Load pretrained components and sampling scheduler
# ============================== #
# VAE model
vae = AutoencoderKL.from_pretrained(
    repo_id, subfolder="vae", use_safetensors=True)

# Tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    repo_id, subfolder="tokenizer")

# Text encoder
text_encoder = CLIPTextModel.from_pretrained(
    repo_id, subfolder="text_encoder", use_safetensors=True)

# Conditional noise model
unet = UNet2DConditionModel.from_pretrained(
    repo_id, subfolder="unet", use_safetensors=True)


# schedulers = [
#     DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler"),
#     DPMSolverSinglestepScheduler.from_pretrained(repo_id, subfolder="scheduler"),
#     DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler"),
#     DEISMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler"),
#     UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler"),
#     PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler"),
# ]
schedulers = [
    DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler"),

    DPMSolverSinglestepScheduler(
        algorithm_type="dpmsolver++",
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        # clip_sample=False,
        dynamic_thresholding_ratio=0.995,
        lower_order_final=True,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        sample_max_value=1.0,
        # set_alpha_to_one=False,
        # skip_prk_steps=True,
        solver_order=1,
        solver_type="midpoint",
        # steps_offset=1,
        thresholding=False,
        use_karras_sigmas=False,
    ),

    DPMSolverMultistepScheduler(
        algorithm_type="dpmsolver++",
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        # clip_sample=False,
        dynamic_thresholding_ratio=0.995,
        lower_order_final=True,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        sample_max_value=1.0,
        # set_alpha_to_one=False,
        # skip_prk_steps=True,
        solver_order=1,
        solver_type="midpoint",
        steps_offset=1,
        thresholding=False,
        timestep_spacing="linspace",
        use_karras_sigmas=False,
    ),

    DEISMultistepScheduler(
        algorithm_type="deis",
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        # clip_sample=False,
        dynamic_thresholding_ratio=0.995,
        lower_order_final=True,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        sample_max_value=1.0,
        # set_alpha_to_one=False,
        # skip_prk_steps=True,
        solver_order=1,
        solver_type="logrho",
        steps_offset=1,
        thresholding=False,
        timestep_spacing="linspace",
        use_karras_sigmas=False,
    ),
]
# ============================== #


# Device
# ============================== #
device = "cuda"
vae.to(device)
text_encoder.to(device)
unet.to(device)
# ============================== #


# Create text embedding
# ============================== #
PROMPTS = [
    # "A lion standing in a boat flowing along a milky river, "
    # "a lot of flowers on the river bank, cartoon style, "
    # "high quality, detailed image",
    #
    # "A glowing tank is running in the dark black sky full of stars, "
    # "vivid color, high quality, detailed photograph",

    "A medieval ship in the middle of a big storm with "
    "thunderbolts striking down, gloomy, vibrant colors, "
    "photographic image, high quality, detailed photograph",

    # "A ruined Gothic castle under the luminous moon, "
    # "high quality, detailed photograph"
]

PROMPT_IDS = [
    # "0",
    # "1",
    "2",
    # "3",
]

assert len(PROMPT_IDS) == len(PROMPTS), \
    f"len(PROMPT_IDS)={len(PROMPT_IDS)} while " \
    f"len(PROMPTS)={len(PROMPTS)}!"

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
guidance_scale = 7.5  # Scale for classifier-free guidance

NUM_INFERENCE_STEPS = 24  # Number of denoising steps
# SEEDS = [20, 21, 73]
SEEDS = [21]
# ============================== #


# Save folder
# ============================== #
PROJECT_DIR = "/home/dkie/Working/Results/Github/" \
            "DeepLearningProjects/SampleProjects/StraightFlowODE"
if not os.path.exists(PROJECT_DIR):
    os.makedirs(PROJECT_DIR)
# ============================== #


# Text embedding
# ============================== #
batch_size = len(PROMPTS)

# Conditional embedding
# ------------------------- #
text_input = tokenizer(
    PROMPTS, padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True, return_tensors="pt"
)

# transformers.tokenization_utils_base.BatchEncoding
# print(f"type(text_input): {type(text_input)}")

# text_input has the following attributes:
# data: A dictionary containing:
#   - input_ids: A list of ids for tokens in the text
#   - attention_mask: Mask over input tokens
# print(f"text_input.__dict__: {text_input.__dict__}")

with torch.no_grad():
    text_embeddings = text_encoder(
        text_input.input_ids.to(device))[0]
# ------------------------- #

# Unconditional embedding
# ------------------------- #
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length",
    max_length=max_length, return_tensors="pt")

# transformers.tokenization_utils_base.BatchEncoding
# print(f"type(uncond_input): {type(uncond_input)}")

uncond_embeddings = text_encoder(
    uncond_input.input_ids.to(device))[0]
# ------------------------- #

# Final embedding is the combination of unconditional and conditional embeddings
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
# ============================== #

# Create random noise
# ============================== #
latents_by_seed = []

for seed in SEEDS:
    generator = torch.manual_seed(seed)

    latents_ = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )

    latents_by_seed.append(latents_)
# ============================== #


# Generate images
# ============================== #
for scheduler in schedulers:
    print(f"\nUse scheduler {scheduler.__class__.__name__}!")
    print(scheduler)

    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    # exp_dir = join(PROJECT_DIR, "StableDiffusion",
    #                f"{scheduler.__class__.__name__}_Order1", "exp")
    exp_dir = join(PROJECT_DIR, "StableDiffusion",
                   f"{scheduler.__class__.__name__}_Order1")
    if not exists(exp_dir):
        os.makedirs(exp_dir)

    # with open(join(exp_dir, "prompts.txt"), "w") as f:
    #     f.writelines(PROMPTS)

    for seed_idx, latents in enumerate(latents_by_seed):
        print(f"\nGenerate images with seed={SEEDS[seed_idx]}!")

        # Sampling latents
        latents = latents.to(device)

        print(f"Sample latents using {scheduler.__class__.__name__}!")
        for t in tqdm(scheduler.timesteps):

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert latents to images
        print(f"Decode latents into images!")
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            images = vae.decode(latents).sample

        print(f"Save images to files!")
        for prompt_idx, image in enumerate(images):
            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
            image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)

            save_file = join(exp_dir, f"{scheduler.__class__.__name__}_prompt{PROMPT_IDS[prompt_idx]}"
                                      f"_seed{SEEDS[seed_idx]}_N{NUM_INFERENCE_STEPS}.png")
            image.save(save_file)

            # plt.imshow(image)
            # plt.title(f"{scheduler.__class__.__name__}")
            # plt.show()
# ============================== #