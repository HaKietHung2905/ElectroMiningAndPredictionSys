import os

#UPLOAD_FOLDER = 'uploads'

def save_uploaded_image(file, UPLOAD_FOLDER):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    return file_path
