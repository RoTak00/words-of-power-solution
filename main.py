import requests
from time import sleep
import random
import json
import csv
import pandas as pd
import os
import numpy as np

from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

is_p1 = True


# Load word embeddings
embeddings = None 
if os.path.exists('GoogleNews-vectors-negative300-SLIM.bin'):
    embeddings = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin", binary=True)
 
def get_vector(word):
    try:
        return embeddings[word]
    except KeyError:
        return np.zeros(300)
    
    # Load CSV
df = pd.read_csv("word_mappings.csv")

# Encode input and response
if embeddings is not None:
    X = np.stack(df["word"].apply(get_vector))
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(df["response"])
    y_encoded = to_categorical(y_labels)

with open('words.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    labeled_words = {int(k): v for k, v in data.items()}

if embeddings is not None:
    model = load_model('model.keras')


def Catchup(word, labeled_words, history):

    global is_p1

    if len(history) == 0:
        base_idx = len(labeled_words) // 2
        offset = random.randint(-4, 4)
        return labeled_words[max(0, min(len(labeled_words) - 1, base_idx + offset))]

    score_difference = history[-1]['p1_total_cost'] - history[-1]['p2_total_cost']
    if not is_p1:
        score_difference = -1 * score_difference

    if score_difference > 0:
        base_idx = int(len(labeled_words) * 0.75)
    else:
        base_idx = int(len(labeled_words) * 0.25)

    offset = random.randint(-2, 2)
    idx = max(0, min(len(labeled_words) - 1, base_idx + offset))
    return idx, labeled_words[idx]
    


with open('words.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    labeled_words = {int(k): v for k, v in data.items()}

#sort labeled words by cost 
labeled_words = sorted(labeled_words.items(), key=lambda x: x[1]['cost'])

print(labeled_words)

# print(labeled_words)
for idx, entry in labeled_words:
    print(entry)

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5

history = []

def check_exists(word):

    rows = []

    if not os.path.exists('history.csv'):
        return None
    
    history_csv = pd.read_csv('history.csv')
    
    for index, row in history_csv.iterrows():
        if row['word'] == word:
            rows.append(row)

    if len(rows) == 0:
        return None
    else:
        return rows

def find_lowest_winner(rows):

    min_row = None
    for row in rows:
        if row['win'] == 1:
            if min_row is None or row['cost'] < min_row['cost']:
                min_row = row

    return min_row

def find_highest_loser(rows):

    max_row = None
    for row in rows:
        if row['win'] == 0:
            if max_row is None or row['cost'] > max_row['cost']:
                max_row = row

    return max_row

def find_word_id(word):
    for idx, entry in labeled_words:
        if entry['text'] == word:
            return idx
        
def calc_score_difference(history):
    if len(history) == 0:
        return 0
    
    score_difference = history[-1]['p1_total_cost'] - history[-1]['p2_total_cost']
    if not is_p1:
        score_difference = -1 * score_difference

    return score_difference         

def predict_response(input_word, labeled_words, model):
    global history
    vec = get_vector(input_word)
    probs = model.predict(np.array([vec]))[0]
    predicted_text = label_encoder.inverse_transform([np.argmax(probs)])[0]

    sorted_indices = np.argsort(probs)
    for idx in sorted_indices:
        predicted_text = label_encoder.inverse_transform([idx])[0]
        for idx, entry in labeled_words:
            if entry['text'] == predicted_text:
                selected = labeled_words[idx]
                return selected
    return None
        

def play_game(player_id):

    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            print(response.json())
            sys_word = response.json()['word']
            round_num = response.json()['round']

            sleep(1)

        if round_id > 1:
            status = requests.get(status_url)
            print(status.json())

            history_status = status.json()['status']

            id1 = find_word_id(history_status['p1_word'])
            id2 = find_word_id(history_status['p2_word'])

            entry_1 = {'id': id1,
                        'word': history_status['system_word'], 
                       'response': history_status['p1_word'],
                        'cost': history_status['p1_word_cost'],
                         'win': history_status['p1_won'] }
            

            entry_2 = {
                        'id': id2,
                        'word': history_status['system_word'], 
                       'response': history_status['p2_word'],
                        'cost': history_status['p2_word_cost'],
                         'win': history_status['p2_won'] }

            df = pd.DataFrame([entry_1, entry_2])

            df.to_csv("history.csv", mode='a', header=False, index=False)

            history.append(status.json()['status'])

        data = check_exists(sys_word)

        id = None 

        INC_COST = 3
        if data is not None:
            lowest_winner = find_lowest_winner(data)
            
            if lowest_winner is not None:
                id = lowest_winner['id']
                print("Found Old Lowest Winner")
            
            if lowest_winner is None:
                highest_loser = find_highest_loser(data)

                highest_loser_cost = highest_loser['cost']

                new_cost = highest_loser_cost + INC_COST

                # find an item in labeled_words with this cost 
                for idx, entry in labeled_words:
                    if(entry['cost'] == new_cost):
                        id = idx

                        print("Found Old Highest Loser")
                        break
            
            print(id)

        if id is None:

            score_difference = calc_score_difference(history)

            if embeddings is not None and score_difference > 0:
                response = predict_response(sys_word, labeled_words, model)
                print("Prediction")
            
                if response is None:
                    id, word = Catchup(sys_word, labeled_words, history)
                    print("Catchup")
                else:
                    id = response[0]
                    word = response[1]
            else:
                id, word = Catchup(sys_word, labeled_words, history)
                print("Catchup")
            
            print(word)

        if sys_word == 'Mihai':
            id = 1
            print("Mihai")

        print(id)
        data = {"player_id": player_id, "word_id": id, "round_id": round_id}
        
        response = requests.post(post_url, json=data)
        print(response.json())

play_game("ZGoggBEfRu")