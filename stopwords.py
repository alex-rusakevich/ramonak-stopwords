#!/usr/bin/env python
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def contains_any(string, substrings):
    return any(substring in string for substring in substrings)


def main():
    # Создание объекта TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=15_000)

    corpus = []

    for corpus_file in Path("yabc", "data").glob("*.txt"):
        print("Loading '{}'...".format(corpus_file), end=" ")

        with open(corpus_file) as file:
            tsv_file = csv.reader(file, delimiter="\t")

            file_id = None
            file_text = ""

            for line in tsv_file:
                if len(line) != 5 or contains_any(line[4], ("Punc", "Tag")):
                    continue

                if (
                    "NP" in line[4]
                    or not contains_any(
                        line[4],
                        ("Pron", "Prep", "Conj", "Pcle", "Excl", "Mod", "Num", "Adv"),
                    )
                    or re.search(r"\d+", line[2])
                ):
                    file_text += " "
                    file_text += "no_index"
                    continue

                if not file_id:
                    file_id = line[0]

                if line[0] != file_id:
                    file_id = line[0]
                    corpus.append(file_text.strip())
                    file_text = ""

                file_text += " "
                file_text += line[2].lower().strip()

            if file_text:
                corpus.append(file_text.strip())

        print("OK")

    print("Corpuses have been loaded, processing...")

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    vec_names = list(tfidf_vectorizer.get_feature_names_out())

    print("Writing the result to data/stopwords.txt and data/all.csv...")

    f_all = open(DATA_DIR / "stopwords-be-all-scores.csv", "w", encoding="utf8")
    f_sw_sm = open(DATA_DIR / "stopwords-be-sm.txt", "w", encoding="utf8")
    f_sw_md = open(DATA_DIR / "stopwords-be-md.txt", "w", encoding="utf8")
    f_sw_lg = open(DATA_DIR / "stopwords-be-lg.txt", "w", encoding="utf8")
    f_sw_all = open(DATA_DIR / "stopwords-be-all.txt", "w", encoding="utf8")

    tfidf_matrix_sum = np.sum(tfidf_matrix, axis=0).tolist()[0]

    # region Remove no_index, which is always first due to statistics
    pairs = sorted(
        zip(
            tfidf_matrix_sum,
            vec_names,
        ),
        key=lambda x: x[0],
        reverse=True,
    )[1:]
    # endregion

    all_tfidf_sum = sum(i[0] for i in pairs)

    for i, name in pairs:
        if i / all_tfidf_sum >= 0.001:  # Small list
            f_sw_sm.write(name)
            f_sw_sm.write("\n")

        if i / all_tfidf_sum >= 0.0001:  # Medium list
            f_sw_md.write(name)
            f_sw_md.write("\n")

        if i / all_tfidf_sum >= 0.00001:  # Large list
            f_sw_lg.write(name)
            f_sw_lg.write("\n")

        f_sw_all.write(name + "\n")
        f_all.write("{};{}\n".format(i, name))


if __name__ == "__main__":
    main()
