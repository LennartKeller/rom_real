from pathlib import Path
from string import punctuation

import spacy
import pandas as pd

from nltk import sent_tokenize
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import CountVectorizer


SPACY_MODEL = "de_dep_news_trf"
DEVICE = 0

TEXT_COLUMN = "normed_text"
SRC_DATASET = Path("/mnt/data/cophi/JournalClub/rom_real_dataset.csv")
LEMMATIZED_DATASET = Path("../data/rom_real.csv")
DTM_PATH = Path("../data/rom_real_dtm.csv")
TMP_FILE = Path("dtm_temp.txt")

PUNCTUATION = set(punctuation)


def make_lemmatizer(spacy_model: str):

    spacy.require_gpu(DEVICE)
    nlp = spacy.load(spacy_model)

    def lemmatize_text(text: str):
        text = " ".join(text.split())
        sents = sent_tokenize(text, language="german")
        tagged_sents = [nlp(sent) for sent in sents]
        lemmatized_text = " ".join(
            [
                " ".join([token.lemma_.lower() for token in sent])
                for sent in tagged_sents
            ]
        )
        return lemmatized_text

    return lemmatize_text


if __name__ == "__main__":

    if TMP_FILE.exists():
        raise Exception(
            f"Temp-File: {TMP_FILE} already exists. Please remove it manually to proceed!"
        )

    orig_dataset = pd.read_csv(SRC_DATASET)

    pbar = tqdm(orig_dataset[TEXT_COLUMN].to_list())

    lemmatizer = make_lemmatizer(SPACY_MODEL)

    lemmatized_texts = []
    for text in pbar:

        try:
            lemmatized_text = lemmatizer(text)
        except Exception as e:
            print(e)
            lemmatized_text = ""

        with TMP_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{lemmatized_text}\n")

        lemmatized_texts.append(lemmatized_text)

    orig_dataset["lemmatized_text"] = lemmatized_texts
    orig_dataset.to_csv(LEMMATIZED_DATASET, index=False)

   # Make DTM

    count_vec = CountVectorizer(
        tokenizer=lambda text: [
            token.strip("".join(PUNCTUATION))
            for token in text.split(" ")
            if token not in PUNCTUATION
            and not token.strip("".join(PUNCTUATION)).isnumeric()
            and not all([char in PUNCTUATION for char in token])
            and not token.startswith("'")
            and len(token.strip("".join(PUNCTUATION))) > 1
        ],
        max_features=20000
    )

    dtm_numpy = count_vec.fit_transform(lemmatized_texts).todense()
    dtm_numpy = dtm_numpy / dtm_numpy.sum(axis=1)

    vocab = [
        token
        for token, _ in sorted(
            count_vec.vocabulary_.items(), key=lambda entry: entry[1]
        )
    ]

    dtm_df = pd.DataFrame(data=dtm_numpy, columns=vocab)

    titles = orig_dataset.title.apply(lambda title: "_".join(title.split()))
    authors = orig_dataset.author.apply(lambda author: "_".join(author.split()))
    index = titles + "#" + authors

    dtm_df.insert(0, "meta#title_author", index)

    dtm_df.to_csv(DTM_PATH, index=False)
