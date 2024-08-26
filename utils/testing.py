import pandas as pd
from collections import Counter
import re
import time


def load_csv():
    df = pd.read_csv("spotify_million_song_dataset.csv")
    return df


def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def load_data_conv(df):
    songs = []

    for index, song in df.iterrows():
        song_lyrics_paragraphs = song["text"].replace(
            "\n\n", "").split("\n  \n")

        song = []

        for paragraph in song_lyrics_paragraphs:
            cleaned_paragraph = paragraph.replace("\n", "").replace("  ", " ")
            tokenized_paragraph = tokenize(cleaned_paragraph)
            # print(cleaned_paragraph)
            # print("------------------------EOS------------------------")
            song.append(tokenized_paragraph)

        songs.append(song)

    return songs


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
    print(pairs[0])
    return pairs


def load_data(df):
    tally = [0]
    tokens = []
    para_count = 0
    para_prev_length = 0

    for index, song in df.iterrows():
        song_lyrics_paragraphs = song["text"].replace(
            "\n\n", "").split("\n  \n")

        for paragraph in song_lyrics_paragraphs:
            para_count += 1
            tokenized_paragraph = tokenize(
                paragraph.replace("\n", "").replace("  ", " "))
            tokens.extend(tokenized_paragraph)
            para_prev_length += len(tokenized_paragraph)
            tally.append(para_prev_length)

    return tally, tokens


def generate_pairs_fast(tokens, window, tally):
    pairs = []

    for i, value in enumerate(tally):
        if i+1 == len(tally):
            break

        for k in range(tally[i], tally[i+1]):
            localized_k = k - tally[i]
            start = max(0, localized_k - window)
            end = min(tally[i+1] - tally[i], localized_k + window + 1)

            for index in range(start, end):
                if index == localized_k:
                    continue
                else:
                    pairs.append(
                        (tokens[localized_k + tally[i]], tokens[index + tally[i]]))
    # print(pairs[0])
    return pairs


def main1():
    start = time.time()
    df = load_csv()
    tally, tokens = load_data(df)
    pairs = generate_pairs_fast(tokens, 2, tally)
    print("Pairs extracted:", len(pairs))
    end = time.time()

    print("Elapsed time:", end - start)


def main2():
    start = time.time()
    df = load_csv()
    songs = load_data_conv(df)
    pairs = generate_pairs(songs, 2)
    print("Pairs extracted:", len(pairs))
    end = time.time()

    print("Elapsed time:", end - start)


if __name__ == "__main__":
    # main1()
    main2()
