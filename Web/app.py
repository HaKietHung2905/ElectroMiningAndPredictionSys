import os
import pandas as pd
import json
from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template
from functions import Predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    # Add your main function logic here
    return render_template('index.html')

@app.route('/query_form', methods=['POST'])
def query_form():
    if 'csv_input' not in request.files:
        return 'No file part'
    file = request.files['csv_input']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the CSV file using pandas
        df = pd.read_csv(filepath)
        
        # Convert the DataFrame to HTML (for demonstration purposes)
        #data_html = df.to_html()
        #codeType = 'A_A'

        codeType = request.form['selectType']
        print(codeType)
        predictType = request.form['predictType']
        print(predictType)
        
        if predictType == 'hour':
            df  = Predict.predict1Hour(codeType,df)
            Result = df.iloc[0].to_dict()
            return render_template('result4hour.html', result=Result)
        else:
            Result = Predict.predict4Day(codeType,df)            
            return render_template('result.html', result=Result) 
        # Plot the data
        print(Result)
       
        #return Result
    else:
        return 'Invalid file type'

if __name__ == '__main__':
    import pandas as pd
    app.run(debug=True)
