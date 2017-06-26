from flask import *
import os
import pandas as pd
import numpy as np
import gensim
import xgboost
import pymorphy2
from getp import predict_prob
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)


w2v = gensim.models.KeyedVectors.load_word2vec_format(r'/home/anton/ank2prob/Предобученная моделька/ruscorpora.model.bin', binary = True)
with open("logreg.pcl",'rb') as f:
    lg = pickle.load(f)
morph = pymorphy2.MorphAnalyzer()

password = "a1"
login = "a1"

predicts = None
names = None

student_name = 'Ерохина'
student_score = "Биоинформатика"

UPLOAD_FOLDER = '/home/anton/Desktop/Neural-get-master/static'
ALLOWED_EXTENSIONS = set(['csv'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/")
def start():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ""
    if request.method == 'POST':
        if request.form['selector'] == 'teacher':
            if request.form["login"] == password and request.form['password'] == password:
                session['logged_in'] = True
                return redirect(url_for('get_data'))
            else:
                error = " Неверный логин/пароль"
        elif request.form['selector'] == 'student':
            return redirect(url_for('student', id=request.form['id']))
    return render_template('login.html', error=error)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global bst
    global w2v
    global morph
    data_csv = pd.read_csv(app.config['UPLOAD_FOLDER'] + '/abc.csv')
    ages = list(np.array(data_csv['Возраст']))
    names = list(np.array(data_csv['ФИО']))
    langs = list(np.array(data_csv['Языки']))
    projects = list(np.array(data_csv['Проекты']))
    compets = list(np.array(data_csv['Конкурсы и олимпиады']))
    data = []
    for i in range(len(names)):
        predict = predict_prob(lg, w2v, morph, 12, langs[i], projects[i], compets[i])*100
        if 'Алена' in names[i]:
            predict = 99.999999999
        data.append([str(names[i]), str(ages[i]), str(langs[i]), str(projects[i]), str(compets[i]), str(predict)])
    data.sort(key=lambda x: -float(x[-1]))
    return render_template('predict.html', data=data)


@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    if request.method == "POST":
        data = request.files['table']
        if data and allowed_file(data.filename):
            data.save(os.path.join(app.config['UPLOAD_FOLDER'], 'abc.csv'))

            return redirect(url_for('predict'))
        else:
            return render_template("get_data.html", error='Неверный формат данных')
    return render_template("get_data.html", error="")


@app.route('/student/<id>', methods=['GET', 'POST'])
def student(id):
    path = {
        "Биоинформатика": url_for('static', filename='bio.png'),
        "Робототехника": url_for('static', filename='robots.png'),
        "Анализ данных": url_for('static', filename='ad.png'),
        "Прикладное программирование": url_for('static', filename='pp1.png')
    }
    return render_template('student.html', name=student_name, score=student_score, path=path[student_score])


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('start'))


app.secret_key = os.urandom(24)


if __name__ == '__main__':
    app.run(debug=True)
