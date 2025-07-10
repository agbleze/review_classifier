import string

def preprocess_texts_to_tokens(sentence):
    text_tokens = []
    for text in sentence.split(" "):
            text = text.lower()
            if text not in string.punctuation and not text.isnumeric():
                if text[-1] in string.punctuation:
                    text = text[:-1]
                text_tokens.append(text)
    return text_tokens