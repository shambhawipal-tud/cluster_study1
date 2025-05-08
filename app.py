from flask import Flask, render_template, jsonify, request
import os
from information_extraction.clustering_logic import cluster_pipeline

app = Flask(__name__)

# Absolute file paths
CASE_STUDY_PATH = r"C:\Users\shambhawi\Source\Repos\cluster_study1\Case Study-1.txt"
COMPANY_LAWS_PATH = r"C:\Users\shambhawi\Source\Repos\cluster_study1\company_laws.txt"
CASE_STUDY_SUMMARY_PATH = r"C:\Users\shambhawi\Source\Repos\cluster_study1\case_study_summary.txt"


@app.route('/')
def index():
    case_study_text = ""
    case_study_path = CASE_STUDY_SUMMARY_PATH
    try:
        with open(case_study_path, "r", encoding="utf-8") as f:
            case_study_text = f.read()
    except FileNotFoundError:
        case_study_text = "Case study summary not found."
    return render_template("index.html", case_study=case_study_text)

@app.route('/next')
def next_page():
    case_study_text = ""
    company_laws_text = ""
    base = os.path.dirname(__file__)
    try:
        with open(CASE_STUDY_SUMMARY_PATH, "r", encoding="utf-8") as f:
            case_study_text = f.read()
        with open(COMPANY_LAWS_PATH, "r", encoding="utf-8") as f:
            company_laws_text = f.read()
    except FileNotFoundError:
        case_study_text = "Case study summary not found."
        company_laws_text = "Company laws text not found."
    return render_template("next.html",
                           case_study=case_study_text,
                           company_laws=company_laws_text)

@app.route('/toggle')
def toggle_content():
    base = os.path.dirname(__file__)
    try:
        with open(CASE_STUDY_SUMMARY_PATH, "r", encoding="utf-8") as f:
            cs = f.read()
        with open(COMPANY_LAWS_PATH, "r", encoding="utf-8") as f:
            cl = f.read()
    except FileNotFoundError:
        cs, cl = "Not found.", "Not found."
    return jsonify({"case_study": cs, "company_laws": cl})

@app.route('/review-clusters', methods=['GET','POST'])
def review_clusters():
    if request.method == 'POST':
        # load transcript
        base = os.path.dirname(__file__)
        with open(CASE_STUDY_PATH, "r", encoding="utf-8") as f:
            transcript = f.read()
        # Q1
        selected = request.form.getlist('policies')
        user_justifications = {}
        for pol in selected:
            user_justifications[pol] = request.form.get(f'justification_{pol}', '')
        # Q2
        grouping = request.form.get('grouping', 'event')
        if grouping == 'unsure':
            grouping = 'event'

        # run clustering
        clusters = cluster_pipeline(transcript,
                                    user_justifications,
                                    grouping,
                                    n_clusters=5)
        return render_template('review_clusters.html', clusters=clusters)
    else:
        return render_template('review_clusters.html', clusters=None)

if __name__ == '__main__':
    app.run(debug=True)
