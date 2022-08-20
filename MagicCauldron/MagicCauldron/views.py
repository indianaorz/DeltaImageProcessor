"""
Routes and views for the flask application.
"""

from datetime import datetime
from lib2to3.pytree import convert
from flask import render_template
from flask import Flask, request, jsonify
from MagicCauldron import app
from PIL import Image


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

    n = 0.5
    num = 1
    cfg = 10
    iterations = 1
    name = "test"
    prompt = "contents dissolving into a magic cauldron, digital art, high resolution, birds eye view"
    seed = 100
    steps = 100

    init = "img/" + name + ".jpg"

    # data = {"image": str(request.data)}
    # return jsonify(data)

    # Check whether the specified path exists or not
    isExist = os.path.exists("img/")
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("img/")

    convert_and_save(request.data, init)

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

    from ldm.simplet2i import T2I

    model = T2I()

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
    image_string = str(encoded_string);
    image_string = image_string.replace("b\'", "data:image/png;base64,")

    if request.method == "POST":
        data = {
            "image": image_string
        }
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
    z = b[b.find(b"/9") :]
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
