import src.data_process as data_process

import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
import sys
import yaml


def get_model(config):
    model_path = config['inference']['model_path']
    model = keras.models.load_model(model_path)
    return model


def predict_word_language(model, word, config):
    alphabet = config['load']['master_alphabet']
    alphabet_power = len(alphabet)
    max_word_length = config['load']['max_word_length']

    letter_to_number = {alphabet[i]: i for i in range(alphabet_power)}
    embedding = data_process.get_embedding_from_word(word, letter_to_number, max_word_length)
    probs = model.predict(np.array([embedding]))
    return probs


if __name__ == "__main__":
    with open('config/params.yaml', 'rb') as f:
        yaml_config = yaml.safe_load(f.read())  # load the config file
    if yaml_config is None:
        print('config error')
        sys.exit()

    my_model = get_model(yaml_config)
    if my_model is None:
        print('model error')
        sys.exit()

    test_word = yaml_config['inference']['test_word']
    print(f'test word: {test_word}')
    prediction_probs = predict_word_language(my_model, test_word, yaml_config)

    langs = yaml_config['load']['languages']
    for i in range(len(langs)):
        lang = langs[i]
        score = prediction_probs[0][i]
        print(f'{lang}: {str(round(100 * score, 2))}%')
