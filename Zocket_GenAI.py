import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, request, render_template, send_file
from io import BytesIO

# creating Flask app
app = Flask(__name__)

# Loading model and pipe
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# backend routes
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    prompt = request.forms['prompt']
    image = pipe(prompt).images[0]
    img_buffer = BytesIO()
    image.save(img_buffer)
    img_buffer.seek(0)
    return send_file(img_buffer, mimetype='image/png')

if __name__=="__main__":
    app.run(debug=True)