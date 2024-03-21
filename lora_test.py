from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline
import torch


#pipe = DiffusionPipeline.from_pretrained("C:/tool/InstantFace-Conda/checkpoints/sd_xl_base_1.0.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

pipe = StableDiffusionXLPipeline.from_single_file("./checkpoints/sd_xl_base_1.0.safetensors")
pipe.to("cuda")


pipe.load_lora_weights("./lora", weight_name="Cute_3D_Cartoon.safetensors")

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]

images.save("lora_test.jpg")


'''pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("./lora", weight_name="Cute_3D_Cartoon.safetensors")
prompt = "bears, pizza bites"
image = pipeline(prompt).images[0]
image.save("lora.jpg")'''

