import os
from flask import Flask, request, redirect, url_for,send_from_directory
from werkzeug import secure_filename
import worker
from os.path import basename
 
 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            base,ext=os.path.splitext(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            worker.search.delay(os.path.join(app.config['UPLOAD_FOLDER'], filename))
 

            return redirect(url_for('uploaded_file',
                                    filename=filename))
 
    return '''
    <!DOCTYPE html>
    <head></head>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
 

 
 
if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=5000)
