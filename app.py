from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    summary_text = ""
    file_path = os.path.join(os.path.dirname(__file__), "case_study_summary.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            summary_text = file.read()
    except FileNotFoundError:
        summary_text = "Summary file not found."

    return render_template("index.html", summary=summary_text)

@app.route('/next')
def next_page():
    summary_text = ""
    file_path = os.path.join(os.path.dirname(__file__), "case_study_summary.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            summary_text = file.read()
    except FileNotFoundError:
        summary_text = "Summary file not found."

    return render_template("next.html", summary=summary_text)


if __name__ == '__main__':
    app.run(debug=True)
