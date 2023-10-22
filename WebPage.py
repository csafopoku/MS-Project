from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)


# Load your pre-trained machine learning model from a saved pickle file
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    error_message = None


    if request.method == 'POST':
        try:
            # Extract and convert user input data from the web form
            value_eur = float(request.form['value_eur'])
            release_clause_eur = float(request.form['release_clause_eur'])
            age = int(request.form['age'])
            potential = int(request.form['potential'])
            movement_reactions = int(request.form['movement_reactions'])
            defending = int(request.form['defending'])
            wage_eur = float(request.form['wage_eur'])
            mentality_interceptions = int(request.form['mentality_interceptions'])
            attacking_crossing = int(request.form['attacking_crossing'])
            goalkeeping_diving = int(request.form['goalkeeping_diving'])
            goalkeeping_reflexes = int(request.form['goalkeeping_reflexes'])
            defending_marking_awareness = int(request.form['defending_marking_awareness'])
            goalkeeping_positioning = int(request.form['goalkeeping_positioning'])
            mentality_composure = int(request.form['mentality_composure'])
            mentality_penalties = int(request.form['mentality_penalties'])


            # Create a list of feature values from the user input
            feature_values = [value_eur, release_clause_eur, age, potential,
                              movement_reactions, defending, wage_eur, mentality_interceptions,
                              attacking_crossing, goalkeeping_diving, goalkeeping_reflexes,
                              defending_marking_awareness, goalkeeping_positioning,
                              mentality_composure, mentality_penalties]


            # Make a prediction using the machine learning model
            prediction = model.predict([feature_values])[0]
            
            # Average value of the Y_train (overall or rating) was 65.67778716216216
            average_target = np.mean(65.67778716216216)
            
            # Calculate confidence as a percentage with 1 decimal place
            confidence = 100 * (1 - (abs(prediction - average_target) / average_target))

            # Format confidence to have 1 decimal place and add a percentage sign
            confidence = f'{confidence:.1f}%'
        except ValueError as ve:
            error_message = "Invalid input values. Please enter valid numerical values."


        except KeyError as ke:
            error_message = f"Missing or incorrect input field: {str(ke)}"


        except Exception as e:
            error_message = f"An error occurred: {str(e)}"


    if error_message:
        if request.method == 'POST':
            return render_template('home.html', error=error_message)
        else:
            return jsonify({'error': error_message})


    return render_template('home.html', prediction=prediction,confidence=confidence, error=error_message)


if __name__ == '__main':
    app.run(port=5000, debug=True)
