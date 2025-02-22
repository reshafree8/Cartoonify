from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import json
from cartoonify import cartoonify_image

app = Flask(__name__)


from jinja2 import Environment

def file_exists(file_path):
    """Check if a file exists before displaying it."""
    return os.path.exists(file_path)

# Register the function as a filter in Jinja
app.jinja_env.filters['file_exists'] = file_exists


UPLOAD_FOLDER = 'static/uploads/'  # Folder for uploaded images
RESULT_FOLDER = 'static/results/'  # Folder for cartoonified images
PROCESSED_IMAGES_FILE = 'processed_images.json'  # JSON file to track images

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Function to save processed images to JSON
def save_processed_image(uploaded_image, cartoon_image):
    try:
        if os.path.exists(PROCESSED_IMAGES_FILE):
            with open(PROCESSED_IMAGES_FILE, 'r') as file:
                images_list = json.load(file)
        else:
            images_list = []

        images_list.append({"uploaded": uploaded_image, "cartoon": cartoon_image})

        with open(PROCESSED_IMAGES_FILE, 'w') as file:
            json.dump(images_list, file, indent=4)

    except Exception as e:
        print("Error saving processed image:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded file
    uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(uploaded_file_path)

    # Generate cartoonified image
    cartoon_file_path = os.path.join(app.config['RESULT_FOLDER'], f"cartoon_{file.filename}")
    cartoonify_image(uploaded_file_path, cartoon_file_path)

    # Save paths to JSON file
    save_processed_image(f"uploads/{file.filename}", f"results/cartoon_{file.filename}")

    # Redirect to results page
    return redirect(url_for('results', 
                            uploaded_image=f"uploads/{file.filename}", 
                            cartoon_image=f"results/cartoon_{file.filename}"))

@app.route('/delete/<filename>')
def delete_image(filename):
    """Deletes a processed image entry even if the file is missing."""
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)

    if os.path.exists(file_path):
        os.remove(file_path)  # Delete file if found
        print(f"Deleted: {file_path}")  # Debugging
    else:
        print(f"File not found: {file_path}")  # Debugging

    # Remove entry from JSON even if file is missing
    if os.path.exists(PROCESSED_IMAGES_FILE):
        with open(PROCESSED_IMAGES_FILE, 'r') as file:
            images_list = json.load(file)

        # Filter out the missing/deleted image from the list
        images_list = [img for img in images_list if img['cartoon'] != f"results/{filename}"]

        # Save updated list
        with open(PROCESSED_IMAGES_FILE, 'w') as file:
            json.dump(images_list, file, indent=4)

        print("JSON updated!")  # Debugging

    return redirect(url_for('converted_photos'))  # Reload page



@app.route('/results')
def results():
    uploaded_image = request.args.get('uploaded_image')
    cartoon_image = request.args.get('cartoon_image')
    if not uploaded_image or not cartoon_image:
        return "Images not found", 404

    return render_template('results.html', 
                           uploaded_image=uploaded_image, 
                           cartoon_image=cartoon_image)

@app.route('/converted-photos')
def converted_photos():
    """Show all processed images with download options."""
    if os.path.exists(PROCESSED_IMAGES_FILE):
        with open(PROCESSED_IMAGES_FILE, 'r') as file:
            images_list = json.load(file)
    else:
        images_list = []

    return render_template('converted_photos.html', images=images_list)

@app.route('/download/<filename>')
def download_image(filename):
    """Allow users to download processed images."""
    file_path = os.path.join('static/results/', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
