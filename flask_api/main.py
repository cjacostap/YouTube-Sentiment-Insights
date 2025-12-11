import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle


app = Flask(__name__) ## initialize the flask app
CORS(app)  # Enable CORS for all routes


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment
    

##Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name=None, model_version=None, vectorizer_path=None, model_path=None, run_id=None):
    """
    Load model and vectorizer, trying MLflow first, then falling back to local files.
    
    Args:
        model_name: Name of the model in MLflow registry (optional)
        model_version: Version of the model in MLflow registry (optional)
        vectorizer_path: Path to the local vectorizer pickle file
        model_path: Path to the local model pickle file (used as fallback)
        run_id: MLflow run_id to load model directly from (optional, alternative to model_name/version)
    
    Returns:
        tuple: (model, vectorizer)
    """
    mlflow.set_tracking_uri("http://ec2-44-193-25-173.compute-1.amazonaws.com:5000/")
    
    # Try to load from MLflow first
    if model_name and model_version:
        try:
            print(f"Attempting to load model '{model_name}' version '{model_version}' from MLflow Model Registry...")
            client = MlflowClient()
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✓ Model loaded successfully from MLflow Model Registry")
            
            # Load vectorizer from local file
            with open(vectorizer_path, 'rb') as file:
                vectorizer = pickle.load(file)
            print("✓ Vectorizer loaded successfully from local file")
            
            return model, vectorizer
        except Exception as e:
            print(f"⚠ Failed to load from MLflow Model Registry: {e}")
            if run_id:
                print(f"Trying to load directly from run_id '{run_id}'...")
                try:
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.pyfunc.load_model(model_uri)
                    print("✓ Model loaded successfully from MLflow run")
                    
                    with open(vectorizer_path, 'rb') as file:
                        vectorizer = pickle.load(file)
                    print("✓ Vectorizer loaded successfully from local file")
                    
                    return model, vectorizer
                except Exception as run_error:
                    print(f"⚠ Failed to load from MLflow run: {run_error}")
            print("Falling back to local model files...")
    elif run_id:
        try:
            print(f"Attempting to load model from MLflow run_id '{run_id}'...")
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✓ Model loaded successfully from MLflow run")
            
            with open(vectorizer_path, 'rb') as file:
                vectorizer = pickle.load(file)
            print("✓ Vectorizer loaded successfully from local file")
            
            return model, vectorizer
        except Exception as e:
            print(f"⚠ Failed to load from MLflow run: {e}")
            print("Falling back to local model files...")
    
    # Fallback to local files
    if model_path is None:
        model_path = "./lgbm_model.pkl"
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✓ Model loaded successfully from local file")
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        print("✓ Vectorizer loaded successfully from local file")
        
        return model, vectorizer
    except Exception as local_error:
        raise Exception(f"Failed to load models from both MLflow and local files. Local error: {local_error}")

# ## Initialize the model and vectorizer
# Try to load run_id from experiment_info.json if available
try:
    import json
    with open('experiment_info.json', 'r') as f:
        exp_info = json.load(f)
        run_id = exp_info.get('run_id')
        print(f"Found run_id in experiment_info.json: {run_id}")
except Exception as e:
    print(f"Could not load experiment_info.json: {e}")
    run_id = None

model, vectorizer = load_model_and_vectorizer(
    model_name="yt_chrome_plugin_model", 
    model_version="1", 
    vectorizer_path="./tfidf_vectorizer.pkl",
    model_path="./lgbm_model.pkl",  # Fallback path
    run_id=run_id  # Alternative: load directly from run_id if model registry fails
)


# CA 20251210: Commented out the load_model function and vectorizer function to load the model and vectorizer from the model registry and local storage

# def load_model(model_path, vectorizer_path):
#     """Load the trained model."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
        
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)
      
#         return model, vectorizer
#     except Exception as e:
#         raise

# # Initialize the model and vectorizer
# model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")  


# CA 20251210: Devuelve un mensaje simple indicando que la API está activa.
@app.route('/')
def home():
    return "Welcome to our flask api"



# Predice el sentimiento de una lista de comentarios usando el modelo y el vectorizer cargados.
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    # print("i am the comment: ",comments)
    # print("i am the comment type: ",type(comments))
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        # predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


# Predice el sentimiento de una lista de comentarios usando el modelo y el vectorizer cargados, y agrega los timestamps a la respuesta.
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)


# Genera un gráfico de torta (pie chart) con los datos de sentimiento.
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500
    

# Genera una nube de palabras (word cloud) con los comentarios.
@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# Genera un gráfico de tendencia (trend graph) con los datos de sentimiento.
@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# Inicia la aplicación Flask en el puerto 5000 (puede ser cambiado con la variable de entorno PORT).
# Nota: En macOS, el puerto 5000 puede estar ocupado por AirPlay Receiver.
# Si tienes problemas, desactiva AirPlay Receiver en System Settings > General > AirDrop & Handoff
# o usa otro puerto: PORT=5001 python flask_api/main.py
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Default to 5000, can be overridden with PORT env var
    app.run(host='0.0.0.0', port=port, debug=True)
