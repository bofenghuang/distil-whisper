#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Bofeng Huang

"""
Normalize French text.

Adapted from: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
"""

import re
import unicodedata

try:
    from num2words import num2words
except ImportError:
    raise ImportError("Please install text_to_num by `pip install num2words`")

try:
    from text_to_num import alpha2digit
except ImportError:
    raise ImportError("Please install text_to_num by `pip install text2num`")


# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)

    # bh: Category "Mn" stands for Nonspacing_Mark
    """
    # fmt: off
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )
    # fmt: on


# adapted to optionally keep selected symbols
def remove_symbols(s: str, keep=""):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    # fmt: off
    return "".join(
        c
        if c in keep
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKC", s)
    )
    # fmt: on


def roman_to_int(s: str):
    roman_numerals = {
        "I": 1,
        "V": 5,
        "X": 10,
        # 'L': 50,
        # 'C': 100,
        # 'D': 500,
        # 'M': 1000
    }

    result = 0
    prev_value = 0

    for char in reversed(s):
        value = roman_numerals[char]
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value

    return result


# todo: L'heure, L
def detect_and_convert_roman(input_str):
    # roman_pattern = r'\b[IVXLCDM]+\b'
    roman_pattern = r"\b[IVX]+\b"
    matches = re.findall(roman_pattern, input_str)

    for match in matches:
        integer_value = roman_to_int(match)
        input_str = input_str.replace(match, str(integer_value), 1)

    return input_str


class FrenchNumber2TextNormalizer:
    """
    Convert any arabic numbers into spelled-out numbers, while handling:

    - remove any commas
    """

    def __init__(self, lang: str = "fr", converter: str = "cardinal"):
        self.lang = lang
        self.converter = converter

        self.replacers = {
            # ordinal
            r"\b1er\b": "premier",
            r"\b1ère\b": "première",
            r"\b2nd\b": "second",
            r"\b2nde\b": "seconde",
            r"\b2[èe]me\b": "deuxième",
            r"\b3[èe]me\b": "troisième",
            r"\b4[èe]me\b": "quatrième",
            r"\b5[èe]me\b": "cinquième",
            r"\b6[èe]me\b": "sixième",
            r"\b7[èe]me\b": "septième",
            r"\b8[èe]me\b": "huitième",
            r"\b9[èe]me\b": "neuvième",
            # numbers
            # r"(\d),(\d)": r"\1 virgule \2",
            # r"(\d).(\d)": r"\1 point \2",
            # r"(\d)\s?\%": r"\1 pour cent ",
            # r"(?<=\d)\s(?=000)": "",  # 1 000 -> 1000
        }

    def preprocess(self, s: str):
        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        # hotfix: add space between 0 and number, num2word can't convert "07" to "zéro sept"
        # "200" -> "200", "000" -> "0 0 0", "070" -> "0 70"
        while re.search(r"\b0\d+", s):
            s = re.sub(r"(?<=\b0)(\d)", r" \1", s)

        # s = re.sub(r"(?<=\d)[\s\,]+(?=\d{3})", "", s)  # remove space and comma within number (12 200 000)
        # replace "h" in time
        # r"(?<=\d)h(?=\d|\s|$)": r"\1 heures \2",
        # s = re.sub(r"(\d)\s*h\s*(00)", r"\1 heures", s)
        # s = re.sub(r"(\d)\s*h\s*(\d)", r"\1 heures \2", s)
        return s

    def __call__(self, s: str):
        s = self.preprocess(s)

        length_diff = 0
        # NB: for two digit separated alphanum
        # for match in re.finditer(r"[1-9][0-9]*|(?:(?<=[^0-9])|(?<=^))0", s):
        # for match in re.finditer(r"\d+", s):
        for match in re.finditer(r"\b\d+\b", s):
            num_word = num2words(match.group(), lang=self.lang, to=self.converter)
            start, end = match.start() + length_diff, match.end() + length_diff
            s = f"{s[:start]} {num_word} {s[end:]}"
            # +2 espaces
            length_diff += len(num_word) - (end - start) + 2
        return s


class FrenchText2NumberNormalizer:
    def __init__(self, language="fr", ordinal_threshold=0):
        self.language = language
        # todo: default 0 or 3
        self.ordinal_threshold = ordinal_threshold

        self.numbers_before_et = [
            "dix",
            "vingt",
            "trente",
            "quarante",
            "cinquante",
            "soixante",
            "cent",
        ]
        self.pattern_hot_fix_a = re.compile(r"(?<!\b{})(\s+et\s+une?\b)".format(r")(?<!\b".join(self.numbers_before_et)))

    def __call__(self, s: str):

        # hot fix for "et un" -> "1"
        # may introduce some unconverted number words
        # if bool(re.search(r"\bet une?\b", s)):
        #     return s

        # if bool(re.search(r"\w+et\s+une?\w+", s)):
        if bool(self.pattern_hot_fix_a.search(s)):
            splits = self.pattern_hot_fix_a.split(s)
            # print(splits)

            for i in range(len(splits)):
                if bool(re.search(r"^\s+et\s+une?$", splits[i])):
                    continue
                splits[i] = self(splits[i])

            return " ".join(splits)

        # "au moins deux étage" -> "au -2 étages"
        # "Elle n'est cependant plus un point d'arrêt" -> "Elle n'est cependant +1 point d'arrêt"
        s = alpha2digit(s, self.language, signed=False, ordinal_threshold=self.ordinal_threshold)

        # hot fix for "zéro zéro zéro Clermont-Ferrand" -> "00 zéro Clermont-Ferrand"
        s = re.sub(r"0 zéro", "00", s, flags=re.IGNORECASE)

        return s


class FrenchTextNormalizer:
    def __init__(
        self,
        remove_diacritics: bool = False,
        # split_letters: bool = False,
    ):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        # self.split_letters = split_letters

        # French filler words which can be ignored when computed wer
        # self.ignore_patterns = None
        # self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um|ah|bah|beh|ben|eh|euh|hein|hum|mmh|oh|pff)\b"

        self.replacers = {
            # standarize symbols
            r"’|´|′|ʼ|‘|ʻ|`": "'",  # replace special quote
            r"−|‐": "-",  # replace special dash
            # standarize characters (for french)
            r"æ": "ae",
            r"œ": "oe",
            # others
            # r"€": " euro ",
            # r"\$": " dollar ",
            # r"&": " et ",
            # abbreviations
            r"\bm\.(?=\s|$)": "monsieur",
            r"\bM\.(?=\s|$)": "Monsieur",
            r"\bmme(?=\s|$)": "madame",
            r"\bmlle(?=\s|$)": "mademoiselle",
        }

        # todo
        # https://www.fluentu.com/blog/french/how-to-count-in-french/#:~:text=French%20accepts%20both%20hyphens%20and,exception%3A%20million%20is%20never%20hyphenated.
        self.numbers_before_dash = [
            "dix",
            "vingt",
            "trente",
            "quarante",
            "cinquante",
            "soixante",
            "quatre",
            "et",  # trente-et-un
            "cent",
        ]

        self.num2text_normalizer = FrenchNumber2TextNormalizer()
        # self.standardize_spellings = EnglishSpellingNormalizer()
        self.text2num_normalizer = FrenchText2NumberNormalizer()

        # latin chars
        # bh: speechbrain version for "en", "fr", "it", "rw"
        # self.kept_chars = "’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî"
        # bh: lowercased
        # self.kept_chars = "a-zàâäéèêëîïôöùûüÿçñ"
        # bh: all
        # check https://en.wikipedia.org/wiki/List_of_Unicode_characters
        french_chars_lower = "a-zàâäéèêëîïôöùûüÿçñ"
        french_chars_upper = "A-ZÀÂÄÇÈÉÊËÎÏÔÖÙÛÜŸ"
        self.alphabet_chars = french_chars_lower + french_chars_upper
        self.number_chars = "0-9"
        self.kept_chars = self.alphabet_chars + self.number_chars

    def __call__(
        self,
        s: str,
        do_lowercase=True,
        do_ignore_words=False,
        symbols_to_keep="'-",
        do_num2text=True,
        do_text2num=False,
    ):
        assert not (do_num2text and do_text2num)

        if do_lowercase:
            s = s.lower()

        # s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        # s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        # s = re.sub(r"^\s*#\d{3}\s", "", s)  # remove beginning http response code

        if self.ignore_patterns is not None and do_ignore_words:
            s = re.sub(self.ignore_patterns, "", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = self.clean(s, keep=symbols_to_keep)  # remove any other markers, symbols, punctuations with a space
        s = s.lower() if do_lowercase else s  # do anther lowercase after normalization (e.g., ℂ -> C)

        # if self.split_letters:
        #     s = " ".join(regex.findall(r"\X", s, regex.U))

        # s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        # s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = self.num2text_normalizer(s) if do_num2text else s  # convert numbers to words
        s = self.text2num_normalizer(s) if do_text2num else s  # convert words to numbers
        # s = self.standardize_spellings(s)

        # todo: I forgot why I did it again here
        # s = re.sub(rf"[^{self.kept_chars}\' ]", "", s)  # remove unnecessary alphabet characters
        s = re.sub(rf"[^{self.kept_chars}{re.escape(symbols_to_keep)}\s]", " ", s)  # remove unnecessary alphabet characters

        # standardize apostrophe
        s = re.sub(r"\s+'", "'", s)  # remove space before an apostrophe
        s = re.sub(r"'\s+", "'", s)  # remove space after an apostrophe
        # s = re.sub(rf"([{self.kept_chars}])\s+'", r"\1'", s)  # standardize when there's a space before an apostrophe
        # s = re.sub(rf"([{self.kept_chars}])'([{self.kept_chars}])", r"\1' \2", s)  # add an espace after an apostrophe
        # s = re.sub(rf"(?<!aujourd)(?<=[{self.kept_chars}])'(?=[{self.kept_chars}])", "' ", s)  # add an espace after an apostrophe (except aujourd'hui)

        # standardize dash
        # this is adapted to zaion annotation protocole but perhaps not general case
        # s = re.sub(r"(?<!\b{})-(?=\S)".format(r")(?<!\b".join(self.numbers_before_dash)), " ", s)  # remove dash not after numbers
        s = re.sub(r"(?:(?<=\s)|(?<=^))\-+(?=\D)", " ", s)  # remove beginning dash in words (not before digit)
        s = re.sub(r"(?<=\w)\-+(?=\s|$)", " ", s)  # remove trailing dash in words
        # s = re.sub(r"(?:^|\s)\-+\w*", " ", s)  # remove words with beginning dash
        # s = re.sub(r"\w*\-+(?=\s|$)", " ", s)  # remove words with trailing dash

        s = re.sub(r"(?:(?<=\s)|(?<=^))['-]+(?=\s|$)", " ", s)  # remove standalone apostrophes or dashes

        # standardize other symbols
        # standardize "?!:"
        if special_symbols := "".join(re.findall(r"[?!:]", s)):
            s = re.sub(rf"([{self.kept_chars}])([{re.escape(special_symbols)}])", r"\1 \2", s)  # add space before symbols
            s = re.sub(rf"([{re.escape(special_symbols)}])([{self.kept_chars}])", r"\1 \2", s)  # add space between symbols and chars
        # standardize ",."
        if special_symbols := "".join(re.findall(r"[,.]", s)):
            s = re.sub(rf"\s+([{re.escape(special_symbols)}])", r"\1", s)  # remove space before symbols
            s = re.sub(rf"([{re.escape(special_symbols)}])([{self.alphabet_chars}])", r"\1 \2", s)  # add space between symbols and non-number chars

        # s = re.sub(rf"[{re.escape(symbols_to_keep)}]{{2,}}", " ", s)  # remove consecutive symbols

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        # s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        # s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space

        return s


if __name__ == "__main__":
    import sys

    raw_text = " ".join(sys.argv[1:])
    print(raw_text)
    print(FrenchTextNormalizer()(raw_text))
