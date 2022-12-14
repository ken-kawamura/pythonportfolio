from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import os
import shutil
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ['マテオ・コバチッチ', 'ルカ・モドリッチ', 'イバン・ペリシッチ']
n_class = len(labels)
img_size = 32
n_result = 3  # 上位3つの結果を表示

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])

    def __call__(self, img, phase='train'):
        return self.base_transform(img)

transform = ImageTransform(size, mean, std)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像読み込み
        img = Image.open(filepath)   # [高さ][幅][色RGB]
        
        img_transformed = transform(img)  # torch.Size([3, 224, 224])
        inputs = img_transformed.unsqueeze_(0)  #torch.Size([1, 3, 224, 224])

        # 予測
        net = models.vgg16()
        net.classifier[6] = nn.Linear(in_features=4096, out_features=3)
        net.load_state_dict(torch.load("../model.pth", map_location=torch.device("cpu")))
        net.eval()  # 評価モード

        y = net(inputs)
        y = F.softmax(y, dim=1)[0]
        sorted_idx = torch.argsort(-y)  # 降順でソート
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i].item()
            ratio = y[idx].item()
            label = labels[idx]
            result += "<p>" + str(round(ratio*100, 1)) + "%の確率で" + label + "です。</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
