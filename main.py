import re
import pandas as pd
import numpy as np
from utils.stop_words import stop_words
import wandb
import pickle as pkl

wandb.login(key="faa50964fc72148fde282009303887184b48d61f")

epochs = 1
lr = 0.01

run = wandb.init(
    project="Song_sentiment_analyzer",
    config={
        "learning_rate": lr,
        "epochs": epochs
    },
)


def load_csv():
    df = pd.read_csv("spotify_million_song_dataset.csv", nrows=15000)
    return df


def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def load_data(df):
    tokenized_songs = []
    total_tokens = 0

    for index, song in df.iterrows():
        song_lyrics_paragraphs = song["text"].replace(
            "\n\n", "").split("\n  \n")

        tokenized_song = []

        for paragraph in song_lyrics_paragraphs:
            cleaned_paragraph = paragraph.replace("\n", "").replace("  ", " ")
            tokenized_paragraph = tokenize(cleaned_paragraph)
            tokenized_filtered_paragraph = []

            for token in tokenized_paragraph:
                if token not in stop_words:
                    tokenized_filtered_paragraph.append(token)
                    total_tokens += 1

            tokenized_song.append(tokenized_filtered_paragraph)

        tokenized_songs.append(tokenized_song)

    print("Total number of words without stop words:", total_tokens)
    return tokenized_songs


def mapping(tokenized_songs):

    word_to_id = {}
    id_to_word = {}
    all_tokens_cache = set()

    for tokenized_song in tokenized_songs:
        for tokenized_para in tokenized_song:
            for token in tokenized_para:
                all_tokens_cache.add(token)

    for index, word in enumerate(all_tokens_cache):
        word_to_id[word] = index
        id_to_word[index] = word

    return word_to_id, id_to_word


def one_hot_encode(id, vocab_size):
    base_vector = [0] * vocab_size
    base_vector[id] = 1
    return base_vector


def generate_pairs(tokenized_songs, window):
    pairs = []

    for tokenized_song in tokenized_songs:
        for tokenized_paragraph in tokenized_song:
            for k, word in enumerate(tokenized_paragraph):
                start = max(0, k - window)
                end = min(len(tokenized_paragraph), k + window + 1)

                for index in range(start, end):
                    if index == k:
                        continue
                    else:
                        pairs.append(
                            (tokenized_paragraph[k], tokenized_paragraph[index]))
    return pairs


def create_batches(pairs, batch_size):

    mini_batches = []
    total_examples = len(pairs)
    num_of_batches = total_examples // batch_size
    leftover = total_examples % batch_size

    for i in range(num_of_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        mini_batch = pairs[start_index:end_index]
        mini_batches.append(mini_batch)

    if leftover > 0:
        last_mini_batch = pairs[-leftover:]
        mini_batches.append(last_mini_batch)

    return mini_batches


def encode_data(batch, vocab_size, word_to_id):
    X = []
    y = []
    for pair in batch:
        X.append(one_hot_encode(word_to_id[pair[0]], vocab_size))
        y.append(one_hot_encode(word_to_id[pair[1]], vocab_size))

    return np.asarray(X), np.asarray(y)


def softmax(x):
    t = np.exp(x)
    t_sum = np.sum(t, axis=1, keepdims=True)
    a = t / t_sum
    return a


def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)


def train(mini_batches, word_to_id):
    W1 = np.random.randn(len(word_to_id), 300) * \
        (np.sqrt(2. / len(word_to_id)))
    W2 = np.random.randn(300, len(word_to_id)) * \
        (np.sqrt(2. / 300))
    try:
        for epoch in range(epochs):
            for index, batch in enumerate(mini_batches):
                X, y = encode_data(batch, len(word_to_id), word_to_id)
                Z1 = np.dot(X, W1)
                Z2 = np.dot(Z1, W2)
                A2 = softmax(Z2)

                loss = cross_entropy(A2, y)
                print(
                    f"Epoch: {epoch+1} | Batch: {index+1}/{len(mini_batches)} | Loss: {loss:.2f}")
                wandb.log({
                    "loss": loss
                })

                dZ2 = A2 - y
                dW2 = np.dot(Z1.T, dZ2)
                temp = np.dot(dZ2, W2.T)
                dW1 = np.dot(X.T, temp)

                W1 -= lr * dW1
                W2 -= lr * dW2
            with open('w1.pkl', 'wb') as f:
                pkl.dump(W1, f)
            with open('w2.pkl', 'wb') as f:
                pkl.dump(W2, f)
    except KeyboardInterrupt:
        with open('w1.pkl', 'wb') as f:
            pkl.dump(W1, f)
        with open('w2.pkl', 'wb') as f:
            pkl.dump(W2, f)


def main():
    df = load_csv()
    tokenized_songs = load_data(df)
    word_to_id, id_to_word = mapping(tokenized_songs)
    pairs = generate_pairs(tokenized_songs, 2)
    mini_batches = create_batches(pairs, 3000)

    train(mini_batches, word_to_id)


if __name__ == "__main__":
    main()
