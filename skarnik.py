import asyncio
import datetime
import itertools
import zipfile
from pathlib import Path

import nltk
import pytz
import requests
from bs4 import BeautifulSoup

nltk.download("stopwords")

from nltk.corpus import stopwords  # noqa: E402


async def fetch_translations(word: str) -> list[str]:
    print("Translating '{}'...".format(word))

    r = requests.get(
        "https://www.skarnik.by/search?term={}&lang=rus".format(word),
        allow_redirects=True,
    )

    soup = BeautifulSoup(r.content, features="html.parser")

    all_words = []

    for font_element in soup.select('font[color="831b03"]'):
        word_list = font_element.text

        for word in word_list.split(","):
            word = word.strip().lower()
            all_words.append(word)

    return tuple(set(all_words))


def zip_results():
    Path("./dist/").mkdir(parents=True, exist_ok=True)

    tz = pytz.timezone("Europe/Minsk")
    minsk_now = datetime.datetime.now(tz)
    file_date_mark = f"{minsk_now:%Y%m%d_%H%M%S}"

    file_name = "./dist/STOPWORDS_{}.zip".format(file_date_mark)

    print("Packing the results into '{}'...".format(file_name), end=" ")

    with zipfile.ZipFile(
        file_name,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zip_ref:
        zip_ref.write("./build/belarusian.txt", arcname="belarusian.txt")

    print("OK")


async def main():
    Path("./build/").mkdir(parents=True, exist_ok=True)

    translations = []

    russian_words = stopwords.words("russian")

    tasks = []
    for word in russian_words:
        task = fetch_translations(word)
        tasks.append(task)

    # планируем одновременные вызовы
    translations = await asyncio.gather(*tasks)
    translations = list(set(itertools.chain(*translations)))
    translations = sorted(
        filter(lambda x: " " not in x, translations),
        key=lambda x: x.replace("і", "и").replace("ў", "щ"),
    )
    translations = sorted(translations, key=lambda x: len(x))

    Path("./build/belarusian.txt").write_text("\n".join(translations), encoding="utf8")

    zip_results()


if __name__ == "__main__":
    asyncio.run(main())
