from flask import Flask, render_template, request, abort, redirect, url_for
import pandas as pd
import nltk, re
from nltk.corpus import stopwords
import subprocess as sp
from subprocess import call
import csv, os, cgi
import cgitb; cgitb.enable()
from werkzeug.utils import secure_filename
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'
app.config['UPLOAD_EXTENSIONS'] = ['.csv']

lr, acc_lr, prec_lr, rec_lr, f1_lr, tfidf_vectorizer_lr, cm_lr = joblib.load('model-development/lr1.pkl')
svm, acc_svm, prec_svm, rec_svm, f1_svm, tfidf_vectorizer_svm, cm_svm = joblib.load('model-development/svm1.pkl')
detr, acc_dt, prec_dt, rec_dt, f1_dt, tfidf_vectorizer_dt,cm_dt = joblib.load('model-development/dt1.pkl')
nb, acc_nb, prec_nb, rec_nb, f1_nb, tfidf_vectorizer_nb, cm_nb = joblib.load('model-development/nb1.pkl') 
stacking, acc_stacking, prec_stacking, rec_stacking, f1_stacking, tfidf_vectorizer, cm_stacking = joblib.load('model-development/stacking1.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST' and ('predict' in request.form or 'file' in request.files):
        stemmer_factory = StemmerFactory()
        ina_stemmer = stemmer_factory.create_stemmer()
        alay_dict = pd.read_csv('static/kamus/new_kamusalay.csv', names = ['original', 'replacement'], encoding='latin-1')
        alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
        nltk.download('stopwords')
        stopwords_ind = stopwords.words('indonesian')
        def remove_stopword(text) :
            clean_text = []
            text = text.split()
            for i in text :
                if i not in stopwords_ind:
                    clean_text.append(i)
            return " ".join(clean_text)           
        def casefolding(text):
            text = text.lower()
            text = re.sub('\\+n', ' ', text)
            text = re.sub('\n'," ",text)
            text = re.sub('rt',' ',text)
            text = re.sub('RT',' ',text)
            text = re.sub('user',' ',text)
            text = re.sub('USER', ' ', text)
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
            text = re.sub(':', ' ', text)
            text = re.sub(';', ' ', text)
            text = re.sub('@',' ',text)
            text = re.sub('\\+n', ' ', text)
            text = re.sub('\n'," ",text)
            text = re.sub('\\+', ' ', text)
            text = re.sub('  +', ' ', text)
            text = re.sub(r'[-+]?[0-9]+', '', text)
            return text      
        def normalize_alay(text):
            return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])
        def remove_emoticon_byte(text):
            text = text.replace("\\", " ")
            text = re.sub('x..', ' ', text)
            text = re.sub(' n ', ' ', text)
            text = re.sub('\\+', ' ', text)
            text = re.sub('  +', ' ', text)
            return text
        def stemming(text):
            text = ina_stemmer.stem(text)
            return text
        def preprocess(text):
            text = casefolding(text)
            text = remove_stopword(text)
            text = remove_emoticon_byte(text)
            text = normalize_alay(text)
            text = stemming(text)
            return text

        def process_lr(text):
            text = tfidf_vectorizer_lr.transform([text])
            pred_lr = lr.predict(text)
            result = ''
            if pred_lr[0] == 0:
                return 'BUKAN ujaran kebencian'
            else:
                return 'Ujaran kebencian'
        
        def process_svm(text):
            text = tfidf_vectorizer_svm.transform([text])
            pred_svm = svm.predict(text)
            result = ''
            if pred_svm[0]== 0:
                return 'BUKAN ujaran kebencian'
            else:
                return 'Ujaran kebencian'
        
        def process_dt(text):
            text = tfidf_vectorizer_dt.transform([text])
            pred_dt = detr.predict(text)
            result = ''
            if pred_dt[0] == 0:
                return 'BUKAN ujaran kebencian'
            else:
                return 'Ujaran kebencian'
        
        def process_nb(text): 
            text = tfidf_vectorizer_nb.transform([text])
            pred_nb = nb.predict(text)
            result = ''
            if pred_nb[0] == 0:
                return 'BUKAN ujaran kebencian'
            else:
                return 'Ujaran kebencian'
        
        def process_stacking(text):
            text = tfidf_vectorizer.transform([text])
            pred_stacking = stacking.predict(text)
            result = ''
            if pred_stacking[0] == 0:
                return 'BUKAN ujaran kebencian'
            else:
                return 'Ujaran kebencian'

        if request.form['predict'] =='' and request.files['file'].filename =='':
            return render_template('index.html', pesan='Please input text or file!!')

        elif request.form['predict'] !='' and request.files['file'].filename !='':
            return render_template('index.html', pesan='Choose one method!!')

        elif 'predict' in request.form and request.form['predict']!='':
            if os.path.exists('uploads/data.csv'):
                os.remove('uploads/data.csv')
            message = request.form['predict']
            message = message.split('.')
            message_dict = {'Tweet': message}
            df = pd.DataFrame(message_dict)
            df.to_csv('uploads/data.csv', index = False, encoding='latin-1')
            data = pd.read_csv('uploads/data.csv', encoding='latin-1')
            data['StopwordRemoval'] = data['Tweet'].apply(remove_stopword)
            data['Casefolding'] = data['Tweet'].apply(casefolding)
            data['Normalize'] = data['Tweet'].apply(normalize_alay)
            data['RemoveEmoticon'] = data['Tweet'].apply(remove_emoticon_byte)
            data['Stemming'] = data['Tweet'].apply(stemming)
            data['HasilPreprocess'] = data['Tweet'].apply(preprocess)
            data['HasilProcess_lr'] = data['Tweet'].apply(process_lr)
            data['HasilProcess_svm'] = data['Tweet'].apply(process_svm)
            data['HasilProcess_dt'] = data['Tweet'].apply(process_dt)
            data['HasilProcess_nb'] = data['Tweet'].apply(process_nb)
            data['HasilProcess_stacking'] = data['Tweet'].apply(process_stacking)
            data.to_csv('uploads/dataset.csv')
            return redirect(url_for('dashboard'))

        elif 'file' in request.files:
            uploaded_file = request.files['file']
            if os.path.exists('uploads/data.csv'):
                os.remove('uploads/data.csv')
            filename = secure_filename(uploaded_file.filename)
            if filename != '':
                file_ext = os.path.splitext(filename)[1]
                if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                    abort(400)
                uploaded_file.filename = "data.csv"
                uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], uploaded_file.filename))

            data = pd.read_csv('uploads/data.csv', encoding='latin-1')

            data['StopwordRemoval'] = data['Tweet'].apply(remove_stopword)
            data['Casefolding'] = data['Tweet'].apply(casefolding)
            data['Normalize'] = data['Tweet'].apply(normalize_alay)
            data['RemoveEmoticon'] = data['Tweet'].apply(remove_emoticon_byte)
            data['Stemming'] = data['Tweet'].apply(stemming)
            data['HasilPreprocess'] = data['Tweet'].apply(preprocess)
            data['HasilProcess_lr'] = data['Tweet'].apply(process_lr)
            data['HasilProcess_svm'] = data['Tweet'].apply(process_svm)
            data['HasilProcess_dt'] = data['Tweet'].apply(process_dt)
            data['HasilProcess_nb'] = data['Tweet'].apply(process_nb)
            data['HasilProcess_stacking'] = data['Tweet'].apply(process_stacking)

            data.to_csv('uploads/dataset.csv')
            return redirect(url_for('dashboard'))

        
        else:
            return "Unsupported Request Method"
    
@app.route('/dashboard.html', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'GET':
        return render_template("dashboard.html")
    elif request.method == 'POST':
        if request.form['dropdown'] =='1':
            if os.path.exists('uploads/dataset.csv'):
                model = "Logistic Regression"
                df = pd.read_csv('uploads/dataset.csv')
                df = df[['Tweet','HasilProcess_lr']]
                count = len(df)
                tweet = df['Tweet']
                hasil = df['HasilProcess_lr']
                return render_template('dashboard.html', model = model, df=df,tweet=tweet, hasil=hasil, count=count)
            else :
                return render_template('dashboard.html') 
        elif request.form['dropdown']=='2':
            if os.path.exists('uploads/dataset.csv'):
                model = "Support Vector Machine"
                df = pd.read_csv('uploads/dataset.csv')
                df = df[['Tweet','HasilProcess_svm']]
                count = len(df)
                tweet = df['Tweet']
                hasil = df['HasilProcess_svm']
                return render_template('dashboard.html', model=model, df=df, tweet=tweet, hasil=hasil, count=count)
            else :
                return render_template('dashboard.html')
        elif request.form['dropdown'] =='3':
            if os.path.exists('uploads/dataset.csv'):
                model = "Naive Bayes (Bernoulli)"
                df = pd.read_csv('uploads/dataset.csv')
                df = df[['Tweet','HasilProcess_nb']]
                count = len(df)
                tweet = df['Tweet']
                hasil = df['HasilProcess_nb']
                return render_template('dashboard.html',model = model, df=df, tweet=tweet, hasil=hasil, count=count)
            else :
                return render_template('dashboard.html')
        elif request.form['dropdown']=='4':
            if os.path.exists('uploads/dataset.csv'):
                model = "Decision Tree"
                df = pd.read_csv('uploads/dataset.csv')
                df = df[['Tweet','HasilProcess_dt']]
                count = len(df)
                tweet = df['Tweet']
                hasil = df['HasilProcess_dt']
                return render_template('dashboard.html',model = model, df=df, tweet=tweet, hasil=hasil, count=count)
            else :
                return render_template('dashboard.html')
        elif request.form['dropdown']=='5':
            if os.path.exists('uploads/dataset.csv'):
                model = "Stacking (semua model)"
                df = pd.read_csv('uploads/dataset.csv')
                df = df[['Tweet','HasilProcess_stacking']]
                count = len(df)
                tweet = df['Tweet']
                hasil = df['HasilProcess_stacking']
                return render_template('dashboard.html',model = model, df=df, tweet=tweet, hasil=hasil, count=count)
            else :
                return render_template('dashboard.html')
        else:
            return render_template('dashboard.html')
    else:
        return "Unsupported Request Method"
    

@app.route('/casefolding.html')
def casefolding():
    if os.path.exists('uploads/dataset.csv'):
        df = pd.read_csv('uploads/dataset.csv')
        df = df[['Tweet','Casefolding']]
        count = len(df)
        return render_template('casefolding.html', df=df, count=count)
    else :
        return render_template('casefolding.html')

@app.route('/normalize.html')
def cleaning():
    if os.path.exists('uploads/dataset.csv'):
        df = pd.read_csv('uploads/dataset.csv')
        df = df[['Tweet','Normalize']]
        count = len(df)
        return render_template('normalize.html', df=df, count=count)
    else :
        return render_template('normalize.html')

@app.route('/datalatih.html')
def datalatih():
    return render_template('datalatih.html')

@app.route('/datauji.html')
def datauji():
    return render_template('datauji.html')

@app.route('/hasilpreprocess.html')
def hasilpreprocess():
    if os.path.exists('uploads/dataset.csv'):
        df = pd.read_csv('uploads/dataset.csv')
        df = df[['Tweet','HasilPreprocess']]
        count = len(df)
        return render_template('hasilpreprocess.html', df=df, count=count)
    else :
        return render_template('hasilpreprocess.html')

@app.route('/stemming.html')
def stemming():
    if os.path.exists('uploads/dataset.csv'):
        df = pd.read_csv('uploads/dataset.csv')
        df = df[['Tweet','Stemming']]
        count = len(df)
        return render_template('stemming.html', df=df, count=count)
    else :
        return render_template('stemming.html')

@app.route('/stopward-removal.html')
def StopwordRemoval():
    if os.path.exists('uploads/dataset.csv'):
        df = pd.read_csv('uploads/dataset.csv')
        df = df[['Tweet','StopwordRemoval']]
        count = len(df)
        return render_template('stopward-removal.html', df=df, count=count)
    else :
        return render_template('stopward-removal.html')     

@app.route('/removeemoticon.html')
def tokenizing():
    if os.path.exists('uploads/dataset.csv'):
        df = pd.read_csv('uploads/dataset.csv')
        df = df[['Tweet','RemoveEmoticon']]
        count = len(df)
        return render_template('removeemoticon.html', df=df, count=count)
    else :
        return render_template('removeemoticon.html') 

@app.route('/visualisasi.html', methods=['GET', 'POST'])
def visualisasi():
    if request.method == 'GET':
        return render_template("visualisasi.html")
    elif request.method == 'POST':
        if request.form['dropdown'] =='1':
            return render_template('visualisasi.html', model = "Logistic Regression", confussion = cm_lr, akurasi = acc_lr, recall = rec_lr, f1_score = f1_lr, precision=prec_lr)
        elif request.form['dropdown']=='2':
            return render_template('visualisasi.html', model = "Support Vector Machine", confussion = cm_svm, akurasi = acc_svm, recall = rec_svm, f1_score = f1_svm, precision=prec_svm)
        elif request.form['dropdown']=='3':
            return render_template('visualisasi.html', model = "Naive Bayes (Bernoulli)", confussion = cm_nb, akurasi = acc_nb, recall = rec_nb, f1_score = f1_nb, precision=prec_nb)
        elif request.form['dropdown']=='4':
            return render_template('visualisasi.html', model = "Decision Tree", confussion = cm_dt, akurasi = acc_dt, recall = rec_dt, f1_score = f1_dt, precision=prec_dt)
        elif request.form['dropdown']=='5':
            return render_template('visualisasi.html', model = "Stacking (semua model)", confussion = cm_stacking, akurasi = acc_stacking, recall = rec_stacking, f1_score = f1_stacking, precision=prec_stacking)
        else:
            return render_template('visualisasi.html')
    else:
        return "Unsupported Request Method"

if __name__ == '__main__':
  app.run(port=5000,debug=True)