import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image


app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')




@app.route('/')
def home():
	return render_template('home.html')

@app.route('/' , methods=['GET', 'POST'])
def predictt():
    if request.method == "POST":
        msg = request.files["file"]
        msg2 = msg.save(os.path.join("uploads",msg.filename))
        
        
        img = image.load_img(msg, target_size=[64,64])
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
            
    return render_template('home.html', prediction=result)
if __name__ == '__main__':
    app.run(debug=True)