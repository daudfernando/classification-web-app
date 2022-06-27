from flask import Flask, render_template,request,redirect
import pickle
import sklearn
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':

        with open('forest_pickle','rb') as r:
            model = pickle.load(r)
        
        Age = float(request.form['age'])
        Sex = float(request.form['gender'])
        ALB = float(request.form['alb'])
        ALP = float(request.form['alp'])
        ALT = float(request.form['alt'])
        AST = float(request.form['ast'])
        BIL = float(request.form['bil'])
        CHE = float(request.form['che'])
        CHOL = float(request.form['chol'])
        CREA = float(request.form['crea'])
        GGT = float(request.form['ggt'])
        PROT = float(request.form['prot'])

        datas = np.array((Age,Sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT))
        datas = np.reshape(datas, (1,-1))

        isHepatitis = model.predict(datas)

        return render_template('hasil.html', finalData=isHepatitis)
    
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)