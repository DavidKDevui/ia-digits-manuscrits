from flask import Flask, request, jsonify, render_template, render_template_string, flash
import base64
from werkzeug.utils import secure_filename
from io import BytesIO
import numpy as np
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageOps
import pandas as pd
import openai
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
openai.api_key = "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"


@app.route('/')
def index():
    """
    Retourne la page d'accueil
    ---
    tags:
      - Frontend
    responses:
      200:
        description: La page d'accueil a été retournée avec succès.
    """
    return render_template('index.html')



@app.route('/entrainer-le-modele', methods=['GET'])
def entrainer_un_modele():
    """
    Retourne la page d'entraînement du modèle
    ---
    tags:
      - Frontend
    responses:
      200:
        description: La page d'entraînement du modèle a été retournée avec succès.
    """
    return render_template('entrainer-le-modele.html')



@app.route('/assistance')
def assistance():
    """
    Retourne la page d'assistance
    ---
    tags:
      - Frontend
    responses:
      200:
        description: La page d'assistance a été retournée avec succès.
    """
    return render_template('assistance.html')



@app.route('/model', methods=['GET'])
def query_chatgpt():    
    """
    Interroge le modèle GPT-3.5 pour obtenir une réponse à une question donnée.
    ---
    tags:
      - Backend
    parameters:
      - name: question
        in: query
        type: string
        required: true
        description: La question à poser au modèle.
    responses:
      200:
        description: Une réponse réussie
        schema:
          type: object
          properties:
            response:
              type: string
              example: "Voici la réponse du modèle."
    """
    question = request.args.get('question')
    prompt = f"{question}. Réponds brièvement, en maximum deux phrases"

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
    )

    return completion['choices'][0]['message']['content']
    

@app.route('/training', methods=['POST'])
def training():
    """
    Entraîne le modèle de réseau neuronal avec les données fournies.
    ---
    tags:
      - Backend
    parameters:
      - name: Jeu de données d'entraînement
        in: formData
        type: file
        required: true
        description: Le fichier de données d'entraînement (format CSV).
      - name: Jeu de données de validation
        in: formData
        type: file
        required: true
        description: Le fichier de données de validation (format CSV).
      - name: Epochs
        in: formData
        type: integer
        required: true
        description: Le nombre d'époques d'entraînement.
    responses:
      200:
        description: Entraînement terminé avec succès.
      400:
        description: Requête incorrecte. Fichiers manquants ou données d'entrée invalides.
      500:
        description: Erreur interne du serveur. Une erreur s'est produite pendant l'entraînement.
    """
    try:
        # Vérifier et stocker les fichiers reçus dans des variables
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        epoch = int(request.form['epoch'])

        if not file1 or not file2:
            flash('Missing files')
            template = f'''<p style="color: red">Fichiers manquants !</p>'''
            return render_template_string(template)

        if file1.filename == '' or file2.filename == '':
            template = f'''<p style="color: red">Aucun fichier n'a été uploadé !</p>'''
            return render_template_string(template)

        # Sécuriser les noms de fichiers
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        # Sauvegarder les fichiers
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(file1_path)
        file2.save(file2_path)

        # Lire les fichiers CSV
        df = pd.read_csv(file1_path, header=None)
        df_validation = pd.read_csv(file2_path, header=None)

        # Création séparation des features / target
        y_train = to_categorical(df[0])
        X_train = df.drop([0], axis=1).values.reshape(-1, 28, 28, 1) / 255.0  # Normalisation des données

        y_test = to_categorical(df_validation[0])
        X_test = df_validation.drop([0], axis=1).values.reshape(-1, 28, 28, 1) / 255.0  # Normalisation des données

        # Conception d'un modèle de réseau de neuronne à convolution
        my_model = Sequential()
        my_model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
        my_model.add(MaxPooling2D(pool_size=(2,2)))
        my_model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
        my_model.add(MaxPooling2D(pool_size=(2,2)))
        my_model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu'))
        my_model.add(MaxPooling2D(pool_size=(2,2)))
        my_model.add(Flatten())
        my_model.add(Dense(10, activation='softmax'))

        # Compilation du modèle
        my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrainement du modèle
        my_model.fit(X_train, y_train, epochs=epoch, validation_data=(X_test, y_test))

        # Exportation du modèle
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mnist_model.h5')
        accuracy = my_model.evaluate(X_test, y_test)
        
        my_model.save(model_path)

        template = f'''<p style="color: green">L'entraînement est terminé ! Le modèle est opérationel, avec une précision de {round(accuracy[1] * 100, 1)}%</p><br><a href="/">Tester le modèle</a>'''
    except Exception as e:
        template = f'''<p style="color: red">Une erreur s'est produite lors de l'entraînement du modèle : {str(e)}</p>'''

    return render_template_string(template)

    
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédise le chiffre manuscrit contenu dans l'image fournie.
    ---
    tags:
      - Backend
    parameters:
      - name: image
        in: body
        required: true
        description: L'image du chiffre manuscrit encodée en base64.
        schema:
          type: object
          properties:
            image:
              type: string
    responses:
      200:
        description: Prédiction réussie.
        schema:
          type: object
          properties:
            prediction:
              type: integer
              description: Le chiffre prédit.
      400:
        description: Requête incorrecte. L'image fournie est vide ou invalide.
      500:
        description: Erreur interne du serveur. Le modèle n'a pas encore été entraîné.
    """
    if os.path.exists('uploads/mnist_model.h5'):
        model = load_model('uploads/mnist_model.h5')
        data = request.get_json()["image"]
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data)).convert("L")
        image = ImageOps.invert(image)
        image_resized = image.resize((28, 28), Image.LANCZOS)
        image_array = np.array(image_resized)
        non_black_pixels = np.sum(image_array != 0)
        if non_black_pixels == 0:
            return jsonify({"prediction": "L'image semble être vide. Veuillez fournir une image contenant un chiffre."})
        
        image_array = image_array.reshape(-1, 28, 28, 1)
        prediction = model.predict(image_array)[0]
        predicted_digit = prediction.argmax()
        
        # Retourner la prédiction
        return jsonify({"prediction": int(predicted_digit)})
    else:
        return jsonify({"prediction": "Le modèle n'a pas encore été entrainé"})


if __name__ == '__main__':
    app.run(debug=True)