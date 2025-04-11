from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    case_study_text = ""
    
    case_study_path = os.path.join(os.path.dirname(__file__), "case_study_summary.txt")
    
    try:
        with open(case_study_path, "r", encoding="utf-8") as file:
            case_study_text = file.read()
    except FileNotFoundError:
        case_study_text = "Case study summary not found."

    return render_template("index.html", case_study=case_study_text)

@app.route('/next')
def next_page():
    case_study_text = ""
    company_laws_text = ""
    
    case_study_path = os.path.join(os.path.dirname(__file__), "case_study_summary.txt")
    company_laws_path = os.path.join(os.path.dirname(__file__), "company_laws.txt")
    
    try:
        with open(case_study_path, "r", encoding="utf-8") as file:
            case_study_text = file.read()
        
        with open(company_laws_path, "r", encoding="utf-8") as file:
            company_laws_text = file.read()
            
    except FileNotFoundError:
        case_study_text = "Case study summary not found."
        company_laws_text = "Company laws text not found."

    return render_template("next.html", case_study=case_study_text, company_laws=company_laws_text)

@app.route('/toggle')
def toggle_content():
    case_study_text = ""
    company_laws_text = ""
    
    case_study_path = os.path.join(os.path.dirname(__file__), "case_study_summary.txt")
    company_laws_path = os.path.join(os.path.dirname(__file__), "company_laws.txt")
    
    try:
        with open(case_study_path, "r", encoding="utf-8") as file:
            case_study_text = file.read()
        
        with open(company_laws_path, "r", encoding="utf-8") as file:
            company_laws_text = file.read()
            
    except FileNotFoundError:
        case_study_text = "Case study summary not found."
        company_laws_text = "Company laws text not found."
    
    return jsonify({"case_study": case_study_text, "company_laws": company_laws_text})

if __name__ == '__main__':
    app.run(debug=True)
