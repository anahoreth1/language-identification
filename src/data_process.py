
def get_embedding_from_word(word, letter_to_number, max_word_length):
    alphabet_power = len(letter_to_number)

    embedding_length = alphabet_power * max_word_length
    embedding = [0] * embedding_length
    for n, letter in enumerate(word):
        letter_number = letter_to_number[letter]
        position_in_embedding = n * alphabet_power + letter_number
        embedding[position_in_embedding] = 1
    return embedding
