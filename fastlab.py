from fastapi import FastAPI, Request, Form, HTTPException, File, UploadFile
from typing import List
import hashlib
import numpy as np
import io
import requests
from PIL import Image
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import matplotlib.pyplot as plt
import os
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/image_form", response_class=HTMLResponse)
async def make_image(request: Request):
    return templates.TemplateResponse("forms.html", {"request": request})

@app.post("/image_form", response_class=HTMLResponse)
async def make_image(request: Request,
                     noise_level: float = Form(),
                     files: List[UploadFile] = File(description="Multiple files as UploadFile"),
                     resp: str = Form()):

    recaptcha_secret = "6LftbOgpAAAAAC8YIIB3p2x0s58eEnrzx-5Sw9t3"

    recaptcha_data = {
        'secret': recaptcha_secret,
        'response': resp
    }

    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    recaptcha_verification = requests.post(recaptcha_url, data=recaptcha_data)
    recaptcha_result = recaptcha_verification.json()

    if not recaptcha_result['success']:
        raise HTTPException(status_code=400, detail="Ошибка проверки капчи")

    ready = False
    if len(files) > 0:
        if len(files[0].filename) > 0:
            ready = True

    images = []
    original_histogram_images = []
    noisy_histogram_images = []

    if ready:
        images = ["static/" + hashlib.sha256(file.filename.encode('utf-8')).hexdigest() for file in files]
        content = [await file.read() for file in files]
        p_images = [Image.open(io.BytesIO(con)).convert("RGB") for con in content]

        for i in range(len(p_images)):
            original_histogram = get_histogram(p_images[i])

            noise = np.random.normal(0, noise_level, (p_images[i].size[0],p_images[i].size[1],3))
            noisy_image = np.clip(p_images[i] + np.rot90(noise), 0, 255).astype(np.uint8)

            noisy_histogram = get_histogram(noisy_image)

            image_with_noise = Image.fromarray(noisy_image)
            image_with_noise.save(images[i]+'.jpg')

            original_histogram_image = create_histogram_image(original_histogram)
            noisy_histogram_image = create_histogram_image(noisy_histogram)

            original_histogram_image_path = f"static/original_histogram_{i}.png"
            noisy_histogram_image_path = f"static/noisy_histogram_{i}.png"

            original_histogram_image.save(original_histogram_image_path)
            noisy_histogram_image.save(noisy_histogram_image_path)
            original_histogram_images.append(original_histogram_image_path)
            noisy_histogram_images.append(noisy_histogram_image_path)

        return templates.TemplateResponse("forms.html", {"request": request, "ready": ready,
                                                         "images": [image + '.jpg' for image in images],
                                                         "original_histogram_images": original_histogram_images,
                                                         "noisy_histogram_images": noisy_histogram_images,})

@app.get("/view_image", response_class=HTMLResponse)
async def show_histogram(request: Request):
    view_image = "static/view_image.png"
    return templates.TemplateResponse("view_image.html", {"request": request, "view_image": view_image})

def get_histogram(image):
    pixels = np.array(image)

    r_histogram, r_bins = np.histogram(pixels[:, :, 0].flatten(), bins=np.arange(0, 257, 10))
    g_histogram, g_bins = np.histogram(pixels[:, :, 1].flatten(), bins=np.arange(0, 257, 10))
    b_histogram, b_bins = np.histogram(pixels[:, :, 2].flatten(), bins=np.arange(0, 257, 10))

    max_length = max(len(r_histogram), len(g_histogram), len(b_histogram))
    r_histogram = np.pad(r_histogram, (0, max_length - len(r_histogram)), mode='constant')
    g_histogram = np.pad(g_histogram, (0, max_length - len(g_histogram)), mode='constant')
    b_histogram = np.pad(b_histogram, (0, max_length - len(b_histogram)), mode='constant')

    return r_histogram.tolist(), g_histogram.tolist(), b_histogram.tolist()

def create_histogram_image(histograms):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(np.arange(0, len(histograms[0]) * 10, 10), histograms[0], color='r', width=10, label='Red')
    ax.bar(np.arange(0, len(histograms[1]) * 10, 10), histograms[1], color='g', width=10, label='Green')
    ax.bar(np.arange(0, len(histograms[2]) * 10, 10), histograms[2], color='b', width=10, label='Blue')
    ax.set_title('Гистограмма распределения цветов')
    ax.set_xlabel('Значение пиксилей')
    ax.set_ylabel('Частота')
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)
