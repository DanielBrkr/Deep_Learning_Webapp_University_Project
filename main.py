
# coding=utf-8
"""Main function run to start web server"""
from __future__ import annotations
import os
import shutil
import json
import copy
import time
from PIL import Image
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, SubmitField
from wtforms.validators import InputRequired, ValidationError
from werkzeug.utils import secure_filename
from flask_jsglue import JSGlue
from pathlib import Path
from backend.model_factory import Factory
from frontend import utils


# start flask server
app = Flask(__name__, template_folder='./frontend/templates', static_folder='./frontend/static')
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'frontend/static/temp'
jsglue = JSGlue(app)

DATA_FORMATS = ['jpg', 'jpeg', 'png', 'webp']
MODELS = {"ResNet50V2": "ResNet50V2", "Xception_Transfer": "Xception Transfer", "Classical_ML": "Classical ML"}


def data_types(form, _field):
    """
    Data Format Validation
    """
    for file in form.file.data:
        if file.filename.split('.')[-1] not in DATA_FORMATS:
            raise ValidationError("Wrong Data Format")


class UploadFileForm(FlaskForm):  # html form
    """
    Upload file form
    Attributes
    ----------
    class: FlaskForm
    """
    file = MultipleFileField("File(s) Upload", validators=[InputRequired(), data_types])
    submit = SubmitField("Upload File(s)", id='upload-button1')


class GlobalVariables:
    def __init__(self):
        self.model_name = "ResNet50V2"
        self.model = Factory(self.model_name)
        self.retrained = 0
        self.predicted = False
        self.reset = False

    def set_model(self, m: str):
        self.model_name = m
        self.model = Factory(self.model_name)


gv = GlobalVariables()


def delete_and_create():
    """
    remove 'temp' folder with content if existing (app.config['UPLOAD_FOLDER']
    make a new empty temp folder
    """
    if os.path.isdir(app.config['UPLOAD_FOLDER']):  # check if temp folder exists
        shutil.rmtree(app.config['UPLOAD_FOLDER'])  # remove temp folder if existing
    os.makedirs(app.config['UPLOAD_FOLDER'])  # create new (empty) temp folder


def upload():
    """
    upload function
    """
    successfully = False

    form = UploadFileForm()
    if form.validate_on_submit():
        files_filenames = []
        for file in form.file.data:
            file_filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_filename))
            files_filenames.append(file_filename)
            successfully = True
    elif form.file.data:
        successfully = "format"

    return form, successfully


def get_file_paths():
    temp_file_paths = os.listdir(app.config['UPLOAD_FOLDER'])  # temp rel paths
    for idx, path in enumerate(temp_file_paths):
        temp_file_paths[idx] = os.path.join('../', '/'.join(app.config['UPLOAD_FOLDER'].split("/")[1:]),
                                            path)  # add relevant relative path info
    return temp_file_paths


@app.route("/")
@app.route("/index")
def index():
    """
    view function got start page: index
    """
    height = request.args.get("height") or 0
    width = request.args.get("width") or 0
    gv.retrained = 0
    gv.predicted = False
    gv.reset = False

    if not os.path.isdir(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])

    return render_template("start.html", height=height, width=width)


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/about")
def about():
    """
    view function for page 'about'
    """
    return render_template("test2.html")


@app.route("/test2")
def test2():
    """
    view function for page 'about'
    """
    return render_template("about.html")


@app.route("/shop")
def shop():
    """
    view function for page 'shop'
    """
    return render_template("shop.html")


@app.route("/contact")
def contact():
    """
    view function for page 'contact'
    """
    return render_template("contact.html")


def webp_to_jpeg():
    file_paths = os.listdir('frontend/static/temp')  # temp rel paths
    for idx, path in enumerate(file_paths):
        file_paths[idx] = os.path.join('frontend/static/temp', path)  # add relevant relative path info

    for file_path in file_paths:
        if file_path.endswith('.webp'):
            im = Image.open(file_path)
            im.save(file_path.split('.')[0] + '.jpeg')
            os.remove(file_path)


def clear_folder(folder='frontend/static', to_clear='attention_map'):
    files = os.listdir(folder)
    for file in files:
        if file.startswith(to_clear):
            os.remove(os.path.join(folder, file))


@app.route("/label-upload", methods=["GET", "POST"])
def label_upload():
    """
    view function for page 'label-upload'
    """
    delete_and_create()
    form, successfully = upload()
    webp_to_jpeg()  # always convert webp to jpeg
    clear_folder()  # clear selected files in a selected folder
    gv.predicted = False
    gv.retrained = 0
    gv.reset = False

    return render_template("label-upload.html", form=form, successfully=successfully)


@app.route("/upload_preview", methods=["GET", "POST"])
def upload_preview():
    """
    view function for page 'upload_preview.html'
    """
    model_selection = request.args.get('model-selection')
    if model_selection is not None:
        gv.set_model(model_selection)

    if gv.retrained > 0:
        gv.predicted = False
    temp_file_paths = get_file_paths()

    return render_template("upload_preview.html", images=temp_file_paths)


@app.route("/reset_model", methods=["GET", "POST"])
def reset_model():
    temp_file_paths = get_file_paths()

    model_dict = {"ResNet50V2": 'ResNet50V2', 'Xception_Transfer': 'xception_transfer_model',
                  'Classical_ML': 'ClassicML'}
    root_model_folder = "backend/Trained_Models"
    model_folders = os.path.join(root_model_folder, model_dict[gv.model_name])

    models = os.listdir(model_folders)
    src_path = os.path.join(model_folders, next((s for s in models if 'OG' in s), None))
    dest_path = os.path.join(model_folders, next((s for s in models if 'OG' not in s), None))

    shutil.copy(src_path, dest_path)
    gv.model = Factory(gv.model_name)  # new init
    gv.reset = True
    gv.predicted = False
    print("model reset")

    print(model_folders)
    print(src_path)
    print(dest_path)

    return render_template("upload_preview.html", images=temp_file_paths)


@app.route("/labelling-tool", methods=["GET", "POST"])
def label_tool():
    """
    view function for page 'label_tool'
    """
    temp_file_paths = os.listdir(app.config['UPLOAD_FOLDER'])  # temp rel paths

    # Create an empty JSON file where the results of the current labelling session
    # are stored
    with open('frontend/static/labelling_session.json', 'w', encoding="utf-8") as open_file:
        print("Labelling session json file is created in frontend/static/")
        annotations = []
        # add rel_paths as file_names to labelling_session.json
        for path in temp_file_paths:
            annotations.append({"file_name": path, "label": ""})

        json.dump({"annotations": annotations}, open_file, indent=4)

    for idx, path in enumerate(temp_file_paths):
        temp_file_paths[idx] = os.path.join('../', '/'.join(app.config['UPLOAD_FOLDER'].split("/")[1:]),
                                            path)  # add relevant relative path info

    return render_template("labeling-tool.html", images=temp_file_paths, model_name=MODELS[gv.model_name], retr=gv.retrained)


@app.route("/save_labels", methods=["GET", "POST"])
def save_labels():
    """
    view function for page 'save_labels'
    """
    if request.method == "POST" and os.path.exists('frontend/static/labelling_session.json'):
        print(request.json)
        [dct.pop('certainty', None) for dct in request.json]
        with open('frontend/static/labelling_session.json', 'w', encoding="utf-8") as open_file:
            json.dump({"annotations": request.json}, open_file, indent=4)

        # save to Downloads
        downloads_path = str(Path.home() / "Downloads")
        shutil.copy('frontend/static/labelling_session.json', downloads_path)
        return render_template("labeling-tool.html")

    return render_template("labeling-tool.html")


@app.route("/save_corrected_labels", methods=["GET", "POST"])
def save_corrected_labels():
    """
    view function for page 'save_corrected_labels'
    """
    if request.method == "POST" and os.path.exists('frontend/static/predictions_to_display.json'):
        # 'request.json' is the list of annotations
        print(request.json)
        # Create a deep copy of the annotations to save. This copy will be put into a JSON file 
        # to download with certainties removed.
        annotations_to_download = copy.deepcopy(request.json)
        [dct.pop('certainty', None) for dct in annotations_to_download]
        with open('frontend/static/annotations_to_download.json', 'w', encoding="utf-8") as open_file:
            json.dump({"annotations": annotations_to_download}, open_file, indent=4)
        
        # save to Downloads Folder
        downloads_path = str(Path.home() / "Downloads")
        shutil.copy('frontend/static/annotations_to_download.json', downloads_path)
        
        # Keep the certainties for JSON file 'predictions_to_display.json'.
        with open('frontend/static/predictions_to_display.json', 'w', encoding="utf-8") as open_file:
            json.dump({"annotations": request.json}, open_file, indent=4)

        # Insert corrected labels back into 'labelling_session.json':
        original_annotations = {}
        with open('frontend/static/labelling_session.json', 'r+', encoding="utf-8") as open_file:
            original_annotations = json.load(open_file)
            # For every corrected prediction in the users request...
            for corrected_prediction in request.json:
                # ...find original prediction in 'original_annotations'...
                for original_prediction in original_annotations["annotations"]:
                    # ... where the 'file_name' matches...
                    if original_prediction["file_name"] == corrected_prediction["file_name"]:
                        # ...to then delete the original prediction...
                        original_annotations["annotations"].remove(original_prediction)
                        break
                # ... and append the corrected prediction to the original annotations.
                original_annotations["annotations"].append(corrected_prediction)
        # Open 'labelling_session.json' again to overwrite the content
        with open('frontend/static/labelling_session.json', 'w', encoding="utf-8") as open_file:
            json.dump(original_annotations, open_file, indent=4)

        return render_template("prediction_results.html")

    return render_template("prediction_results.html")


@app.route("/retrained", methods=["GET", "POST"])
def retrained():
    t = 0
    while gv.retrained != 2:
        time.sleep(0.5)
        if t > 10:
            break

    gv.model.show_edge_heatmap(path_save="frontend/static/edge_heatmap.png")

    return render_template("retrained.html")


@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    """
    retrain with the new data
    """

    if request.method == "POST":
        data = {"annotations": []}
        for idx, dct in enumerate(request.json):
            data['annotations'].append(dct)
            if data['annotations'][-1]['label'] == '':
                data['annotations'][-1]['label'] = None
            data['annotations'][-1]["certainty"] = None

        gv.retrained = 1

        # backend interface
        print("retrain")
        gv.model.train_model(data)
        gv.retrained = 2
        gv.predicted = True
        print("Model training finished")

    temp_file_paths = get_file_paths()
    return render_template("prediction_results.html", images_paths=temp_file_paths, model_name=MODELS[gv.model_name],
                           retr=gv.retrained)


# not used
@app.route("/prediction-upload", methods=["GET", "POST"])
def predict():
    """
    view function for prediction step 1: upload
    """
    if not gv.predicted:
        with open("./frontend/static/labelling_session.json") as write:
            data = json.load(write)

        delete_and_create()
        gv.form, gv.successfully = upload()
        gv.predicted = True

    return render_template("prediction-upload.html", form=gv.form, successfully=gv.successfully)


@app.route("/prediction_results", methods=["GET", "POST"])
def prediction_results():
    """
    view function for prediction step 2: show results
    """
    if not gv.predicted or gv.reset:
        data = {'annotations': []}
        image_names = os.listdir('frontend/static/temp')
        for name in image_names:
            data['annotations'].append({'file_name': name, 'label': None, 'certainty': None})
            # Create cnn attention_maps if there is not more than 5 images to predict.
            # These can be displayed later in 'prediction_results.html'.
            if len(image_names) <= 5 and gv.model_name != "Classical_ML":
                img_file_path = "./frontend/static/temp/"+name
                attention_map_img = "attention_map_"+name[0:-5]+".png"
           #     gv.model.show_attention(img_file_path, path_save="frontend/static/"+attention_map_img)

        print("predict")
        print(data)
        predictions = gv.model.predict(data)
        for dct in predictions['annotations']:
            dct['certainty'] = float(dct['certainty'])
        print(predictions)

        with open('frontend/static/labelling_session.json', 'w') as file:
            json.dump(predictions, file, indent=4)

        # Sort according to certainty of predictions in 'labelling_session.json'
        utils.sort_certainty('frontend/static/labelling_session.json')

        # create a copy of 'labelling_session.json' because 'prediction_results.html'
        # needs the copy to show the predictions
        shutil.copy('frontend/static/labelling_session.json', 'frontend/static/predictions_to_display.json')
        gv.predicted = True
        gv.retrained = 0
        gv.reset = False

    # The file paths to the images, which the carousel in 'prediction_results.html' needs
    # must be filled in to 'temp_file_paths' here, because after sorting the predictions
    # the order in the JSON file has diverged from the order in 'UPLOAD_FOLDER'
    temp_file_paths = []
    with open('frontend/static/labelling_session.json', 'r') as file:
        predictions = json.load(file)
        for predicted_image in predictions["annotations"]:
            temp_file_paths.append('../static/temp\\'+predicted_image["file_name"])
    print(temp_file_paths)
    
    return render_template("prediction_results.html", images_paths=temp_file_paths, model_name=MODELS[gv.model_name],
                           retr=gv.retrained)


@app.route("/prediction_results_reload", methods=["GET", "POST"])
def prediction_results_reload():
    '''
    view function for when the user reloads the displayed predictions after including
    or excluding certain predictions
    '''
    # Sort 'labelling_session.json' first.
    utils.sort_certainty('frontend/static/labelling_session.json')
    # The file paths to the images, which the caroussel in 'prediction_results.html' needs
    # must be filled in to 'temp_file_paths' here, because after sorting the predictions
    # the order in the JSON file has diverged from the order in 'UPLOAD_FOLDER'
    temp_file_paths = []
    with open('frontend/static/labelling_session.json', 'r') as file:
        predictions = json.load(file)
        for predicted_image in predictions["annotations"]:
            temp_file_paths.append('../static/temp\\'+predicted_image["file_name"])
    print(temp_file_paths)

    if request.method == "POST":
        print(request.form.get("uncertain"))
        # If user wishes to only display uncertain predictions...
        if request.form.get("uncertain") == "uncertain_only":
            # Load predictions
            with open('frontend/static/labelling_session.json', 'r') as file:
                predictions = json.load(file)
                # Copy 'predictions' to 'predictions_reduced' so that removing
                # the predictions is done on the copy. Otherwise, the following for-loop
                # will skip one predicted image after removal.
                predictions_reduced = copy.deepcopy(predictions)
            print(predictions)
            # Determine 'What is uncertain?'

            # remove image from 'temp_file_paths' if its certainty is above a threshold
            for predicted_image in predictions["annotations"]:
                print(predicted_image)
                if predicted_image["certainty"] > 0.25:
                    temp_file_paths.remove('../static/temp\\'+predicted_image["file_name"])
                    predictions_reduced["annotations"].remove(predicted_image)

            # If there are more than 5 predictions 'uncertain', only show the 5 most
            # uncertain predictions (cut list of annotations from the back until len = 5)
            if len(predictions_reduced["annotations"]) > 5:
                list_overflow = 5 - len(predictions_reduced["annotations"])
                predictions_reduced["annotations"] = predictions_reduced["annotations"][0:list_overflow]
                temp_file_paths = temp_file_paths[0:list_overflow]
                
            with open('frontend/static/predictions_to_display.json', 'w') as file:
                json.dump(predictions_reduced, file, indent=4)

        else:
            # Copy 'labelling_session.json' once again to show all predicted images
            # 'prediction_results.html' needs this copy to show the predictions correctly.
            shutil.copy('frontend/static/labelling_session.json', 'frontend/static/predictions_to_display.json')

    # render 'prediction_results.html' with reduced amount of images
    return render_template("prediction_results.html", images_paths=temp_file_paths, model_name=gv.model_name)


if __name__ == "__main__":
    PORT = int(os.environ.get('PORT', 8888))
    DEBUG = int(os.environ.get('DEBUG', 1))
    app.run(debug=True, host='127.0.0.1', port=PORT)
