from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", text="Welcome to the site!")

@app.route('/next')
def next_page():
    return render_template("next.html", text="Welcome to the site!")

if __name__ == '__main__':
    app.run(debug=True)
