"""
Routes and views for the flask application.
"""

from datetime import datetime
from lib2to3.pytree import convert
from flask import render_template
from flask import Flask, request, jsonify
from MagicCauldron import app
from PIL import Image


from ldm.simplet2i import T2I
model = T2I()


import torch
from diffusers import StableDiffusionPipeline    
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPConfig, PreTrainedModel, PretrainedConfig
class NoSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
  
    @torch.no_grad()
    def forward(self, clip_input, images):
        return images, [False]

NoSafetyChecker.__module__ = StableDiffusionSafetyChecker.__module__

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True, safety_checker=NoSafetyChecker)  

pipe.safety_checker = pipe.safety_checker(PretrainedConfig(**pipe.config))
pipe = pipe.to("cuda")


@app.route("/")
@app.route("/home")
def home():
    """Renders the home page."""
    return render_template(
        "index.html",
        title="Home Page",
        year=datetime.now().year,
    )


@app.route("/image", methods=["POST"])
def process_image():

    import json
    import base64
    import os
    
    from urllib.parse import unquote
    strdata = request.data.decode("UTF-8")
    unquoted = unquote(strdata)
    data = json.loads(unquoted)

    n = float(data["strength"])
    num = 1
    cfg = float(data["cfg"])
    iterations = 1
    name = str(data["name"])
    prompt = str(data["prompt"])
    seed = int(data["seed"])
    steps = int(data["steps"])

    init = "img/" + name + ".jpg"

    # data = {"image": str(request.data)}
    # return jsonify(data)

    # Check whether the specified path exists or not
    isExist = os.path.exists("img/")
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("img/")


    convert_and_save(data["image"], init)

    if cfg == 0:
        cfg = 7.0

    if n == 0:
        n = 0.5

    if n >= 1:
        n = 0.99

    if num > 9:
        num = 9

    if iterations < 1:
        iterations = 1

    if iterations > 1:
        num = 1

    if iterations > 16:
        iterations = 16


    outputs = model.img2img(
        prompt=prompt,
        strength=n,
        init_img=init,
        iterations=num,
        seed=(seed),
        steps=steps,
        cfg_scale=cfg,
    )

    


    # f gets closed when you exit the with statement
    # Now save the value of filename to your database

    # Read the image via file.stream
    # img = Image.open(imgdata.stream)
    from PIL import Image

    encoded_string = ""

    with open(outputs[0][0], "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    print(encoded_string)
    image_string = str(encoded_string)
    image_string = image_string.replace("b'", "data:image/png;base64,")

    if request.method == "POST":
        data = {"image": image_string}
        return jsonify(data)
    else:

        return """<html><body>
        Something went horribly wrong
        </body></html>"""

    return jsonify()



@app.route("/conjure", methods=["POST"])
def create_image():

    import json
    import base64
    import os
    
    from urllib.parse import unquote
    strdata = request.data.decode("UTF-8")
    unquoted = unquote(strdata)
    data = json.loads(unquoted)

    n = float(data["strength"])
    num = 1
    cfg = float(data["cfg"])
    iterations = 1
    prompt = str(data["prompt"])
    seed = int(data["seed"])
    steps = int(data["steps"])

    width = int(data["width"])
    height = int(data["height"])


    if cfg == 0:
        cfg = 7.0

    if n == 0:
        n = 0.5

    if n >= 1:
        n = 0.99

    if num > 9:
        num = 9

    if iterations < 1:
        iterations = 1

    if iterations > 1:
        num = 1

    if iterations > 16:
        iterations = 16
        
    width = int(width/64) * 64
    height = int(height/64) * 64

    from torch import autocast
    import torch
    
    generator = torch.Generator("cuda").manual_seed(seed)

    # prompt = "a photograph of an astronaut riding a horse"
    with autocast("cuda"):
      image = pipe(prompt,width=width,height=height,num_inference_steps=steps,guidance_scale=cfg, generator=generator)["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

    # Now to display an image you can do either save it such as:
    image.save(f"name.png")



    #outputs = model.txt2img(
    #    prompt=prompt,
    #    iterations=num,
    #    seed=(seed),
    #    steps=steps,
    #    cfg_scale=cfg,
    #    width=width,
    #    height=height
    #)

    # f gets closed when you exit the with statement
    # Now save the value of filename to your database

    # Read the image via file.stream
    # img = Image.open(imgdata.stream)
    from PIL import Image

    encoded_string = ""

    with open(f"name.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    print(encoded_string)
    image_string = str(encoded_string)
    image_string = image_string.replace("b'", "data:image/png;base64,")

    if request.method == "POST":
        data = {"image": image_string}
        return jsonify(data)
    else:

        return """<html><body>
        Something went horribly wrong
        </body></html>"""

    return jsonify()

def convert_and_save(b64_string, path):
    import base64
    import io
    
    b = b64_string
    enc = b.encode()
    z = enc[enc.find(b"/9") :]
    im = Image.open(io.BytesIO(base64.b64decode(z))).save(path)


@app.route("/contact")
def contact():
    """Renders the contact page."""
    return render_template(
        "contact.html",
        title="Contact",
        year=datetime.now().year,
        message="Your contact page.",
    )


@app.route("/about")
def about():
    """Renders the about page."""
    return render_template(
        "about.html",
        title="About",
        year=datetime.now().year,
        message="Your application description page.",
    )



@app.route("/reveal", methods=["POST"])
def reveal_image():

    import json
    import base64
    import os
    
    from urllib.parse import unquote
    strdata = request.data.decode("UTF-8")
    unquoted = unquote(strdata)
    data = json.loads(unquoted)


    name = str(data["name"])
    
    init = "img/" + name + ".jpg"

    # data = {"image": str(request.data)}
    # return jsonify(data)

    # Check whether the specified path exists or not
    isExist = os.path.exists("img/")
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("img/")


    convert_and_save(data["image"], init)
    
    image_path_or_url = init#"/content/download (9).png" #@param {type:"string"}

    #@markdown 

    #@markdown #####**CLIP models:**

    #@markdown For [StableDiffusion](https://stability.ai/blog/stable-diffusion-announcement) you can just use ViTL14<br>
    #@markdown For [DiscoDiffusion](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb) and 
    #@markdown [JAX](https://colab.research.google.com/github/huemin-art/jax-guided-diffusion/blob/v2.7/Huemin_Jax_Diffusion_2_7.ipynb) enable all the same models here as you intend to use when generating your images

    ViTB32 = False #@param{type:"boolean"}
    ViTB16 = False #@param{type:"boolean"}
    ViTL14 = True #@param{type:"boolean"}
    ViTL14_336px = False #@param{type:"boolean"}
    RN101 = False #@param{type:"boolean"}
    RN50 = False #@param{type:"boolean"}
    RN50x4 = False #@param{type:"boolean"}
    RN50x16 = False #@param{type:"boolean"}
    RN50x64 = False #@param{type:"boolean"}

    models = []
    if ViTB32: models.append('ViT-B/32')
    if ViTB16: models.append('ViT-B/16')
    if ViTL14: models.append('ViT-L/14')
    if ViTL14_336px: models.append('ViT-L/14@336px')
    if RN101: models.append('RN101')
    if RN50: models.append('RN50')
    if RN50x4: models.append('RN50x4')
    if RN50x16: models.append('RN50x16')
    if RN50x64: models.append('RN50x64')

    if str(image_path_or_url).startswith('http://') or str(image_path_or_url).startswith('https://'):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    thumb = image.copy()
    thumb.thumbnail([blip_image_eval_size, blip_image_eval_size])
    display(thumb)

    x = interrogate(image, models=models)
    
    data = {"caption": x}
    return jsonify(data)


import clip
import gc
import numpy as np
import os
import pandas as pd
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from IPython.display import display
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

blip_image_eval_size = 384
blip_model_url = 'C:\\Users\\Lee\\FFCO\\web\\DeltaImageProcessor\\MagicCauldron\\models\\model__base_caption.pth'
#https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'        
blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit='base')
blip_model.eval()
blip_model = blip_model.to(device)

def generate_caption(pil_image):
    gpu_image = transforms.Compose([
        transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]

def load_list(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def rank(model, image_features, text_array, top_count=1):
    top_count = min(top_count, len(text_array))
    text_tokens = clip.tokenize([text for text in text_array]).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = torch.zeros((1, len(text_array))).to(device)
    for i in range(image_features.shape[0]):
        similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
    similarity /= image_features.shape[0]

    top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)  
    return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

def interrogate(image, models):
    caption = generate_caption(image)
    if len(models) == 0:
        print(f"\n\n{caption}")
        return

    table = []
    bests = [[('',0)]]*5
    for model_name in models:
        print(f"Interrogating with {model_name}...")
        model, preprocess = clip.load(model_name)
        model.cuda().eval()

        images = preprocess(image).unsqueeze(0).cuda()
        with torch.no_grad():
            image_features = model.encode_image(images).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        ranks = [
            rank(model, image_features, mediums),
            rank(model, image_features, ["by "+artist for artist in artists]),
            rank(model, image_features, trending_list),
            rank(model, image_features, movements),
            rank(model, image_features, flavors, top_count=3)
        ]

        for i in range(len(ranks)):
            confidence_sum = 0
            for ci in range(len(ranks[i])):
                confidence_sum += ranks[i][ci][1]
            if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                bests[i] = ranks[i]

        row = [model_name]
        for r in ranks:
            row.append(', '.join([f"{x[0]} ({x[1]:0.1f}%)" for x in r]))

        table.append(row)

        del model
        gc.collect()
    display(pd.DataFrame(table, columns=["Model", "Medium", "Artist", "Trending", "Movement", "Flavors"]))

    flaves = ', '.join([f"{x[0]}" for x in bests[4]])
    medium = bests[0][0][0]
    if caption.startswith(medium):
        print(f"\n\n{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}")
        return "\n\n{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"
    else:
        print(f"\n\n{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}")
        return f"\n\n{caption}, {medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"

data_path = "clip-interrogator/data/"

artists = load_list(os.path.join(data_path, 'artists.txt'))
flavors = load_list(os.path.join(data_path, 'flavors.txt'))
mediums = load_list(os.path.join(data_path, 'mediums.txt'))
movements = load_list(os.path.join(data_path, 'movements.txt'))

sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
trending_list = [site for site in sites]
trending_list.extend(["trending on "+site for site in sites])
trending_list.extend(["featured on "+site for site in sites])
trending_list.extend([site+" contest winner" for site in sites])