import csv
import json
import pandas as pd


def cut_context(row):
    return row[:config['context_length']] if config['differ_uppercase'] else row[:config['context_length']].lower()


def process_data(data):
    alphabet = {k: 0 for k in config['alphabet']}

    cutted_data = data.apply(cut_context)
    documents_matrix = []
    for idx, row in enumerate(cutted_data):

        row_matrix = []
        for char in row:
            copy_alphabet = alphabet.copy()
            if char in copy_alphabet.keys():
                copy_alphabet[char] = 1
            row_matrix.append(list(copy_alphabet.values()))

        # append empty vectors to pad rest of the context
        if len(row) < config['context_length']:
            for i in range(config['context_length'] - len(row)):
                row_matrix.append([0] * len(alphabet.keys()))

        documents_matrix.append(row_matrix)
        # print(f'm x l0 : {len(row_matrix[0])} x {len(row_matrix)}')
        if idx % 1000 == 0:
            print(f'{idx} done')
    print(f'Saving ...')
    with open('data.tsv', 'w', encoding='utf8') as tsv_file:
        # tsv_writer = csv.writer(tsv_file, delimiter='', lineterminator='\n')
        for idx, doc in enumerate(documents_matrix):
            # print('\t'.join(str(','.join(str(v) for v in vec)) for vec in doc))
            tsv_file.write('\t'.join(str(','.join(str(v) for v in vec)) for vec in doc) + '\n')
            if idx % 1000 == 0:
                print(f'{idx} done')


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)['data_processing']

    for data_file in config['data_files']:
        corpus = pd.read_csv(data_file)
        data = corpus[config['data_column']]
        process_data(data)
