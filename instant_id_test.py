import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


# prepare models under ./checkpoints
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

#base_model = 'wangqixun/YamerMIX_v8'
base_model = 'stabilityai/stable-diffusion-xl-base-1.0'

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.cuda()

# load Lora
# lora_safetensors_path = "./lora/Cute_3D_Cartoon.safetensors"
pipe.load_lora_weights("./lora/", weight_name="Cute_3D_Cartoon.safetensors")
pipe.fuse_lora(lora_scale=0.85)

# load adapter
pipe.load_ip_adapter_instantid(face_adapter)

# prepare face emb
face_image = load_image("./examples/wim_close.png")
face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
face_emb = face_info['embedding']

print(face_emb)

# prepare face kps
face_pose = load_image("./examples/wim_far.png")
face_info_p = app.get(cv2.cvtColor(np.array(face_pose), cv2.COLOR_RGB2BGR))
face_info_p = sorted(face_info_p, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
face_kps = draw_kps(face_pose, face_info_p['kps'])

# prompt
prompt = "cute easter male bunny"
negative_prompt = ""

# generate image
images = pipe(
    num_images_per_prompt=4,
    prompt=prompt,
    negative_prompt=negative_prompt,
    image_embeds=face_emb,
    image=face_kps,
    controlnet_conditioning_scale=0.7,
    ip_adapter_scale=0.1,
    guidance_scale=7.5,
    num_inference_steps=35
).images 

grid = image_grid(images, rows=2, cols=2)
#image.save("instant_id_test.jpg")

grid.show()