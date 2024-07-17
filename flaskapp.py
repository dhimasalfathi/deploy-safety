from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    Response,
    jsonify,
    request,
    send_file,
    session,
    url_for,
)

# FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm

from wtforms import SubmitField, StringField, DecimalRangeField, IntegerRangeField
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os


# Required to run the YOLOv8 model
import cv2
import logging

logging.basicConfig(level=logging.DEBUG)

# YOLO_Video is the python file which contains the code for our object detection model
# Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import ensure_output_dir, video_detection

app = Flask(__name__)

app.config["SECRET_KEY"] = "muhammadmoin"
app.config["UPLOAD_FOLDER"] = "static/files"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    # We store the uploaded video file path in the FileField in the variable file
    # We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    # video when prompted to do so
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x=""):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)

        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def generate_frames_web(path_x):
    # Create a temporary output path for the webcam stream
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "webcam_output.mp4")
    yolo_output = video_detection(path_x, output_path)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    session.pop("processed_video_path", None)  # Clear the session
    return render_template("upload_video.html")


# Rendering the Webcam Rage
# Now lets make a Webcam page for the application
# Use 'app.route()' method, to render the Webcam page at "/webcam"


@app.route("/webcam")
def webcam():
    return render_template("layout.html", active_page="webcam")


@app.route("/FrontPage", methods=["GET", "POST"])
def front():
    form = UploadFileForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file = form.file.data
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                session["video_path"] = file_path
                flash("File uploaded successfully!", "success")
                return redirect(url_for("front"))
            else:
                flash("No file selected", "error")
        else:
            flash("Form validation failed", "error")
    # Debugging information
    app.logger.info(f"Form errors: {form.errors}")
    app.logger.info(f"Request method: {request.method}")
    app.logger.info(f"Form data: {request.form}")
    app.logger.info(f"Files: {request.files}")

    return render_template("upload_video.html", form=form)


@app.route("/process_video", methods=["POST"])
def process_video():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        ensure_output_dir(output_path)

        # Process the video and save it
        for _ in video_detection(input_path, output_path):
            pass  # We're not yielding frames here, just processing the entire video

        # Store the output path in session for later use
        session["processed_video_path"] = output_path

        flash("Video processed successfully", "success")
        return redirect(url_for("front"))


@app.route("/video")
def video():
    processed_video_path = session.get("processed_video_path")
    if not processed_video_path or not os.path.exists(processed_video_path):
        return "No video available", 404

    def generate():
        cap = cv2.VideoCapture(processed_video_path)
        while True:
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/download_video")
def download_video():
    processed_video_path = session.get("processed_video_path")
    if not processed_video_path or not os.path.exists(processed_video_path):
        return "Processed video not found", 404

    return send_file(processed_video_path, as_attachment=True)


# To display the Output Video on Webcam page
@app.route("/webapp")
def webapp():
    # return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(
        generate_frames_web(path_x=0),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)
