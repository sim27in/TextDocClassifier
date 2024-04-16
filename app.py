import os
import matplotlib.pyplot as plt
from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename
from wtforms import SubmitField, TextAreaField
from predict import TextClassifier
from read import extract_text_from_pdf, extract_text_from_doc
from visualization import generate_bar_graph

app = Flask(__name__)

app.config["SECRET_KEY"] = "abcd1234"
app.config["FILE_FOLDER"] = "static/file"
app.config["IMAGE_FOLDER"] = "static/images"

MODEL_PATH = "model.h5"
ENCODER_PATH = "label_encoder.joblib"

for folder in [app.config["FILE_FOLDER"], app.config["IMAGE_FOLDER"]]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class UploadFileForm(FlaskForm):
    file = FileField("File")
    text = TextAreaField("Text")
    submit = SubmitField("Predict")

text_classifier = TextClassifier(MODEL_PATH, ENCODER_PATH)


@app.route("/", methods=["GET", "POST"])
def upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        if form.file.data and form.text.data:
            return "Please choose only one option for prediction"
        elif form.file.data:
            file = form.file.data
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension == "pdf":
                text = extract_text_from_pdf(file)
            elif file_extension == "doc":
                text = extract_text_from_doc(file)
            else:
                return f"Unsupported File Type {file_extension}"

            input_file_path = os.path.join(app.config["FILE_FOLDER"], secure_filename(file.filename))
            file.save(input_file_path)

        elif form.text.data:
            text = form.text.data

        else:
            return "No file or text provided for prediction."


        predicted_class, probabilities = text_classifier.predict(text)
        class_labels = text_classifier.label_encoder.classes_
        output_image_path = app.config["IMAGE_FOLDER"]
        output_image_file = os.path.join(output_image_path, "bar_graph.png")

        generate_bar_graph(class_labels, probabilities, output_image_file)

        return render_template("predict.html", predicted_class=predicted_class)

    return render_template("home.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)