import csv
from flask import Flask, render_template,request,redirect,url_for,session
import diseaseprediction

app = Flask(__name__)
app.secret_key = '8078'

with open('templates/Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['age'] = request.form['age']
        session['gender'] = request.form['gender']
        return redirect('/symptoms')
    return render_template('index.html')

@app.route('/symptoms', methods=['GET', 'POST'])
def dropdown():
        return render_template('default.html', symptoms=symptoms)

@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    # age = 
    # gender = 

    selected_symptoms = []
    if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom1'])
    if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])

    dic,info,src,disease,symptoms = diseaseprediction.dosomething(selected_symptoms)
    return render_template('disease_predict.html',age=session.get('age'),gender=session.get('gender'),data = dic,info=info,src=src,disease=disease,symptoms = symptoms)

if __name__ == '__main__':
    app.run(debug=True)