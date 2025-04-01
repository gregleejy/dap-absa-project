def update_index(df):
  df['id'] = df['id'].astype(str)
  i = 0
  prev_id = float(df.loc[i, 'id'])
  for j in range(len(df)):
    id = float(df.loc[j, 'id'])
    if(id < prev_id):
      print("test")
      i += 1
    df.loc[j,'id'] = df.loc[j,'id'] + f'_{i}'
    prev_id = id


def find_sublist_indices(lst, sublst, start = 0):
    for i in range(start,len(lst) - len(sublst) + 1):
        if lst[i:i+len(sublst)] == sublst:
            return i  # Return the start index of the match
    return -1  # Not found

def to_token(tokenizer, sentence):
    MAX_LEN = 512
    word_pieces = tokenizer.tokenize(sentence)
    # aspect = tokenizer.tokenize(aspect)
    if len(word_pieces) > MAX_LEN-2:
        word_pieces = word_pieces[:MAX_LEN-2]

    word_pieces = ["[CLS]"] + word_pieces + ["[SEP]"]
    return word_pieces

def word_to_tag(tags, sentence, aspect, polarity, starting_index):
    polarity_tags = [-1]*len(sentence)
    aspect_list = aspect.split(" ")
    sub_len = len(aspect_list)
    ending_index = 0
    for i in range(starting_index, len(sentence) - sub_len + 1):
        if sentence[i:i + sub_len] == aspect_list:
            tags[i] = 1
            if polarity == 'negative':
                polarity_tags[i] = 0
                polarity = 0
            elif polarity == 'neutral':
                polarity_tags[i] = 1
                polarity = 1
            else:
                polarity_tags[i] = 2
                polarity = 2
            for j in range(1, len(aspect_list)):
                tags[i+j] = 2
                polarity_tags[i+j] = polarity
                ending_index = i+j
            break
    return tags, polarity_tags, ending_index


def word_to_tag_tokenize(tags, sentence, aspect_list, polarity, starting_index):
    polarity_tags = [-1]*len(sentence)
    sub_len = len(aspect_list)
    ending_index = 0
    for i in range(max(1, starting_index), len(sentence) - sub_len + 1):
        if sentence[i:i + sub_len] == aspect_list:
            tags[i] = 1
            if polarity == 'negative':
                polarity_tags[i] = 0
                polarity = 0
            elif polarity == 'neutral':
                polarity_tags[i] = 1
                polarity = 1
            else:
                polarity_tags[i] = 2
                polarity = 2
            for j in range(1, len(aspect_list)):
                tags[i+j] = 2
                polarity_tags[i+j] = polarity
                ending_index = i+j
            break
    return tags, polarity_tags, ending_index

def word_to_tag_df_tokenize(df, tokenizer):
    previous_id = 0
    sentence = df.iloc[0]['words']
    sentence = sentence.replace("'", "").strip("][").split(', ')
    sentence = ' '.join(sentence)
    word_pieces = to_token(tokenizer,sentence)
    tags = [0]*len(word_pieces)
    starting_index = 0
    starting_aspect = df.loc[0]['aspect']
    for i in range(len(df)):
        sentence = df.iloc[i]['words']
        sentence = sentence.replace("'", "").strip("][").split(', ')
        sentence = ' '.join(sentence)
        word_pieces = to_token(tokenizer,sentence)
        id = df.iloc[i]['id']
        aspect = df.iloc[i]['aspect']
        #add tags into the dataframe, and reset starting_index and tags if is a new id
        if (id != previous_id):
            for j in df[df['id'] == previous_id].index.tolist():
                df.loc[j,'Tags'] = str(tags)
            starting_index = 0
            tags = [0]*len(word_pieces)
            previous_id = id

        #reset starting index when it is a new aspect
        if (starting_aspect != aspect):
          starting_index = 0
          starting_aspect = aspect

        aspect_token = tokenizer.tokenize(aspect)
        df.loc[i,'aspect_token'] = str(aspect_token)
        df.loc[i, 'words_token'] = str(word_pieces)
        tags,polarity_tag,ending_index = word_to_tag_tokenize(tags,word_pieces,aspect_token,df.iloc[i]['polarity'],starting_index)
        starting_index = ending_index
        df.loc[i,'Polarities'] = str(polarity_tag)

    #adding tags to the last entry in the dataframe
    id = df.iloc[len(df)-1]['id']
    for j in df[df['id'] == id].index.tolist():
        df.loc[j,'Tags'] = str(tags)




def tag_to_word(sentence, predictions):
    """
    predictions: list of tags
    sentence: list of words
    """
    terms = []
    for i, word in enumerate(sentence):
        w = None
        if predictions[i] == 1:
            w = word
            for j in range(i+1, len(sentence)):
                if predictions[j] == 2:
                    w += ' ' + sentence[j]
                else:
                    terms.append(w)
                    i = j
                    break

    return terms

def tag_to_word_df(df, column_name, tags):
    """
    predictions: list of tags
    sentence: list of words
    """
    terms_list = []
    for i in range(len(df)):
        sentence = df.iloc[i]['words_token']
        sentence = sentence.replace("'", "").strip("][").split(', ')
        terms = tag_to_word(sentence, tags[i])
        terms_list.append(terms)
    df[column_name] = terms_list
    return df

def tag_to_pol(sentence, predictions_abte, predictions_absa):
    """
    predictions: list of tags
    sentence: list of words
    """
    terms_pol = {}
    i = 0
    while i < len(sentence):
        word = sentence[i]
        w = None

        if predictions_abte[i] == 1:
            pol_map = {1: 'negative', 2: 'neutral', 3: 'positive'}
            if predictions_absa[i] in pol_map:
                p = pol_map[predictions_absa[i]]
                w = word
                for j in range(i+1, len(sentence)):
                    if predictions_abte[j] == 1:
                        w += ' ' + sentence[j]
                    else:
                        terms_pol[w] = p
                        i = j
                        break
                else:
                    # if inner loop finishes without break
                    terms_pol[w] = p
                    i = len(sentence)
                    break
            else:
                i += 1
        else:
            i += 1

    return terms_pol


def classification_report_read(report_path, encoding):
    """
    Read classification report from file
    """

    with open(report_path, 'r', encoding=encoding) as f:
        report = f.read()
    return report

def print_aligned(report1, report2, title1, title2):
    """
    print two classification report aligned to the columns
    """
    print (1*'\t', title1, 6*'\t', title2)
    report1 = report1.split('\n')
    report2 = report2.split('\n')
    for r1, r2 in zip(report1, report2):
        print(r1, '\t\t', r2)
