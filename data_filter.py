# from pytorch_transformers import *
# from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
# from data import *
# import gensim
# from gensim.models import Word2Vec
# from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm   # Comment this line if using jupyter notebook
from scipy import io as sio
import csv
import argparse
from collections import defaultdict
import os
parser = argparse.ArgumentParser(description='Preprocess ACM Data')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='./data/',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='citation-acm-v8',
                    help='CS, Medical or All: _CS or _Med or (empty)')
# parser.add_argument('--citation_bar', type=int, default=1,
#                     help='Only consider papers with citation larger than (2020 - year) * citation_bar')

args = parser.parse_args()
paper_info = {}
cite_info = defaultdict(lambda: [])
pi = {'abstract':'', 'year':None}
with open(os.path.join(args.data_dir, '%s.txt'%args.domain), 'r', encoding='utf-8') as fin:
    for l in fin:
        l = l.strip('\n')
        if l[:2]=='#*':
            pi['title'] = l[2:].strip()
        elif l[:2] == '#@':
            pi['authors'] = l[2:].strip()
        elif l[:2] == '#t':
            pi['year'] = l[2:].strip()
        elif l[:2] == '#c':
            pi['conf'] = l[2:].strip()
        elif l[:6] == '#index':
            pi['id'] = l[6:]
        elif l[:2] == '#%': #ref
            cite_info[pi['id']].append(l[2:])
        elif l[:2] == '#!': #abstract
            pi['abstract'] = l[2:].strip()
        elif l == '':
            # print(pi)
            paper_info[pi['id']] = pi.strip()
            pi = {'abstract':'', 'year':None}



'''
    use three file to store preprocessed data into tsv file
    paper_info        -> {paper_id, paper_title, paper_abstract, paper_year}
    paper_cite        -> {paper_id, paper_id} done
    paper_venue       -> {paper_id, conf_id}
    conf              -> {conf, conf_id}
    author            -> {author, author_id}
    paper_author_info -> {paper_id, author_id}
'''
author_index_dict = {}
conf_index_dict = {}
for pid in paper_info:
    # print(paper_info[pid])
    try:
        authors = [a.strip() for a in paper_info[pid]['authors'].split(',') if len(a.strip())>3]
        paper_info[pid]['authors'] = authors
        for a in authors:
            if a not in author_index_dict:
                author_index_dict[a] = len(author_index_dict)

        conf = paper_info[pid]['conf']
        if conf not in conf_index_dict:
            conf_index_dict[conf] = len(conf_index_dict)
    except:
        continue



valid_paper = []
with open(os.path.join(args.data_dir, '%s_paper_info.tsv'% args.domain), 'w') as tsvfile:
    for pid in paper_info:
        if 'conf' in paper_info[pid] and 'title' in paper_info[pid]:
            if paper_info[pid]['conf'] in conf_index_dict:
                if 'abstract' not in paper_info[pid]:
                    paper_info[pid]['abstract'] = ''
                tsvfile.write('{}\t{}\t{}\t{}\n'.format(pid, paper_info[pid]['title'], paper_info[pid]['abstract'], paper_info[pid]['year']))
                valid_paper.append(pid)
print('paper info finished')

with open(os.path.join(args.data_dir, '%s_paper_conf.tsv'% args.domain), 'w') as tsvfile:
    for pid in valid_paper:
        tsvfile.write('{}\t{}\n'.format(pid, conf_index_dict[paper_info[pid]['conf']]))
print('paper conf finished')

with open(os.path.join(args.data_dir, '%s_paper_author.tsv'% args.domain), 'w') as tsvfile:
    for pid in valid_paper:
        if 'authors' in paper_info[pid]:
            for author in paper_info[pid]['authors']:
                tsvfile.write('{}\t{}\n'.format(pid, author_index_dict[author]))
print('paper author finished')

with open(os.path.join(args.data_dir, '%s_author.tsv'% args.domain), 'w') as tsvfile:
    for author in author_index_dict:
        if author != '':
            tsvfile.write('{}\t{}\n'.format(author, author_index_dict[author]))
print('author finished')

with open(os.path.join(args.data_dir, '%s_conf.tsv'% args.domain), 'w') as tsvfile:
    for conf in conf_index_dict:
        tsvfile.write('{}\t{}\n'.format(conf, conf_index_dict[conf]))
print('conf finished')


with open(os.path.join(args.data_dir, '%s_paper_cite.tsv'% args.domain), 'w') as tsvfile:
    for pid in valid_paper:
        cited = '\t'.join(cite_info[pid])
        tsvfile.write('{}\t{}\n'.format(pid, cited))
print('paper cite finished')




