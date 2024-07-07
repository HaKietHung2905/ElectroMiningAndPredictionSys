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
        codeType = 'A_A'

        Result  = Predict.getModel(codeType,df)
        
        return Result
    else:
        return 'Invalid file type'


# @app.route('/get_image_paths')
# def get_image_paths():
#     # Retrieve the image paths from the query string
#     image_paths = request.args.get('image_paths')
#     image_paths = json.loads(image_paths)

#     folder_path = 'uploads'  # Adjust this if your folder path is different

#     # Extract image names from the paths
#     image_names = [os.path.basename(path) for path in image_paths]
#     query = request.args.get('query_string')
#     #return image_names
#     #return query
#     return render_template('result.html', folder_path=folder_path, image_names=image_names, query = query)

@app.route('/upload_form', methods=['POST'])
def upload_form():

    if 'image_input' not in request.files:
        return "No file part"
    
    file = request.files['image_input']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    # if file.filename == '':
    #     return "No selected file"
    
    # image_path = saveImages.save_uploaded_image(file, 'static/uploads')

    # instance = saveData.detect_images(image_path)

    # saveData.connect_database('database', instance)

    
    return render_template('index.html')

if __name__ == '__main__':
    import pandas as pd
    app.run(debug=True)
