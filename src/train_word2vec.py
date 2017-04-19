import sys
import codecs
import logging
from gensim.models import Word2Vec

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

def mean(data):
    if len(data) < 1:
        raise ValueError("Can't compute mean of empty list")
    return sum(data)/float(len(data))

def _ss(data):
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def std(data):
    if len(data) < 2:
        raise ValueError("std requires at least two elements")
    ss = _ss(data)
    pvar = ss/float(len(data))
    return pvar**0.5

def find_window_size(fis):
    linelength = []
    with codecs.open(fis, 'r') as fin:
        for line in fin:
            linelength.append(len(line.split()))
    return int(mean(linelength) + std(linelength))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def train_w2v(fishier, wind=111, siz=200):
    with codecs.open(fishier, 'r', 'utf-8') as fin:
        sentences = fin.readlines()
    model = Word2Vec([[wd.strip() for wd in word.split(' ') if wd.strip()] for word in sentences], size=siz, workers=10, window=wind, min_count=1, max_vocab_size=None, sg=1, hs=1)
    logging.info('Saving...')
    model.save_word2vec_format(fishier + '.bin', binary=True)
    logging.info('Done!')

def main():
    try:
        corpus = sys.argv[1]
        logging.info("Corpus file: " + corpus)
        emb_size = int(sys.argv[2])
        logging.info("Embeddings size: " + str(emb_size))
        if len(sys.argv) > 3:
            wind_size = int(sys.argv[3])
        else:
            logging.info("Automatically setting window size...")
            wind_size = find_window_size(corpus)
        logging.info("Window size: " + str(wind_size))
    except Exception, e:
        logging.error(str(e))
        print "----------------------------------------"
        logging.info(sys.argv[0] + "    PATH_TO_CORPUS    EMBEDDING_SIZE    WINDOW_SIZE")
        sys.exit(1)
    print "----------------------------------------"
    train_w2v(corpus, wind_size, emb_size)    

if __name__ == '__main__':
    main()

