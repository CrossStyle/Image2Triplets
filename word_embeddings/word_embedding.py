from transformers import BertTokenizer, BertModel
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def language_model():
    _tokenizer = BertTokenizer.from_pretrained('./chinese_roberta_wwm_ext_pytorch')
    _model = BertModel.from_pretrained('./chinese_roberta_wwm_ext_pytorch')
    return _tokenizer, _model


def get_text(_path):
    all_text = list()
    with open(_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            texts = line.split(' ')
            concat_text = texts[1]
            all_text.append(concat_text)
    return all_text


def get_embeddings(_text, _tokenizer, _model):
    _embeddings = dict()
    for single_text in _text:
        input_ids = torch.tensor([_tokenizer.encode(single_text, add_special_tokens=True)])
        with torch.no_grad():
            _, pooled_output = model(input_ids)
            _embeddings[single_text] = pooled_output
    return _embeddings


def get_vec(_text_set, _embeddings_dict):
    _text_list = list(_text_set)
    _vec = torch.zeros(128, 768)
    for idx, _text in enumerate(_text_list):
        _vec[idx] = _embeddings_dict[_text]
    return _vec, _text_list


def tsne_show(_vec, _text_list):
    embedded_vec = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(_vec)
    plt.figure('Scatter fig')
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(embedded_vec[:, 0], embedded_vec[:, 1])
    plt.show()


def calculate_similarity(_vec, _text_list):
    _s = cosine_similarity(_vec)
    _df = pd.DataFrame(_s)
    _df.columns = _text_list
    _df.index = _text_list
    # plt.figure(1, figsize=(138, 138))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # sns.set(font='SimHei')
    # sns.heatmap(data=_df)
    # plt.show()
    return _df


def export_vec(_path, _dict):
    _vec = torch.zeros(128, 768)
    _labels = []
    with open(_path, 'r', encoding="UTF-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.rsplit('\n')[0].split(' ')[1]
            _labels.append(tmp)
            _vec[idx] = _dict[tmp]
    return _vec, _labels


def export_multiple_vec(_vec, _unseen_index):
    _gzsl_vec = _vec[:13]
    _unseen_vec = _gzsl_vec[torch.tensor([True if i in _unseen_index else False for i in range(13)])]
    _seen_vec = _gzsl_vec[torch.tensor([False if i in _unseen_index else True for i in range(13)])]
    return _seen_vec, _unseen_vec, _gzsl_vec


if __name__ == "__main__":
    tokenizer, model = language_model()
    path = 'index.txt'
    text = get_text(path)
    embeddings = get_embeddings(text, tokenizer, model)
    vec, text_list = get_vec(text, embeddings)
    # tsne_show(vec.numpy(), text_list)
    # df = calculate_similarity(vec.numpy(), text_list)
    final_vec, labels = export_vec(path, embeddings)
    seen_vec, unseen_vec, gzsl_vec = export_multiple_vec(vec, [3, 9, 11])

    torch.save(final_vec, 'vec.pth')
    torch.save(seen_vec, 'seen_vec.pth')
    torch.save(unseen_vec, 'unseen_vec.pth')
    torch.save(gzsl_vec, 'gzsl_vec.pth')

