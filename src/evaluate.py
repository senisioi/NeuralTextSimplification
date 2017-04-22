import sys
import os
import codecs
import logging
from itertools import izip
from SARI import SARIsent
from nltk.translate.bleu_score import *
smooth = SmoothingFunction()
from nltk import word_tokenize

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def folders_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,f)) ]

def files_in_folder_only(mypath):
    return [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def remove_features(sent):
    tokens = sent.split(" ")
    return " ".join([token.split("|")[0] for token in tokens])

def remove_underscores(sent):
    return sent.replace("_", " ")

def replace_parant(sent):
    sent = sent.replace("-lrb-", "(").replace("-rrb-", ")")
    return sent.replace("(", "-lrb-").replace(")", "-rrb-")

def lowstrip(sent):
    return sent.lower().strip()

def normalize(sent):
    return replace_parant(lowstrip(sent))

def as_is(sent):
    return sent

def get_hypothesis(filename):
    hypothesis = '-'
    if "_h1" in filename:
        hypothesis = '1'
    elif "_h2" in filename:
        hypothesis = '2'
    elif "_h3" in filename:
        hypothesis = '3'
    elif "_h4" in filename: 
        hypothesis = '4'
    return hypothesis

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def print_scores(pairs, whichone = ''):
    # replace filenames by hypothesis name for csv pretty print
    for k,v in pairs:
        hypothesis = get_hypothesis(k)
        print "\t".join( [whichone, "{:10.2f}".format(v), k, hypothesis] )

def SARI_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds, refs]]
    scores = []
    for src, pred, ref in izip(*files):
        references = [preprocess(r) for r in ref.split('\t')]
        scores.append(SARIsent(preprocess(src), preprocess(pred), references))
    for fis in files:
        fis.close()
    return mean(scores)


# BLEU doesn't need the source
def BLEU_file(source, preds, refs, preprocess=as_is):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [preds, refs]]
    scores = []
    references = []
    hypothese = []
    for pred, ref in izip(*files):
        references.append([word_tokenize(preprocess(r)) for r in ref.split('\t')])
        hypothese.append(word_tokenize(preprocess(pred)))
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    return corpus_bleu(references, hypothese, smoothing_function=smooth.method3)

def score(source, refs, fold, METRIC_file, preprocess=as_is):
    new_files = files_in_folder(fold)
    data = []
    for fis in new_files:
        # ignore log files
        if ".log" in os.path.basename(fis):
            continue
        logging.info("Processing "+os.path.basename(fis))
        val = 100*METRIC_file(source, fis, refs, preprocess)
        logging.info("Done "+str(val))
        data.append((os.path.basename(fis), val))
    data.sort(key=lambda tup: tup[1])
    data.reverse()
    return data

if __name__ == '__main__':
    try:
        source = sys.argv[1]
        logging.info("Source: " + source)
        refs = sys.argv[2]
        logging.info("References in tsv format: " + refs)
        fold = sys.argv[3]
        logging.info("Directory of predictions: " + fold)
    except:
        logging.error("Input parameters must be: " + sys.argv[0] 
            + "    SOURCE_FILE    REFS_TSV (paste -d \"\t\" * > reference.tsv)    DIRECTORY_OF_PREDICTIONS")
        sys.exit(1)

    '''
        SARI can become very unstable to small changes in the data.
        The newsela turk references have all the parantheses replaced
        with -lrb- and -rrb-. Our output, however, contains the actual
        parantheses '(', ')', thus we prefer to apply a preprocessing
        step to normalize the text.
    '''
    sari_test = score(source, refs, fold, SARI_file, normalize)
    bleu_test = score(source, refs, fold, BLEU_file, lowstrip)

    whichone = os.path.basename(os.path.abspath(os.path.join(fold, '..'))) + \
                    '\t' + \
                    os.path.basename(refs).replace('.ref', '').replace("test_0_", "")
    print_scores(sari_test, "SARI\t" + whichone)
    print_scores(bleu_test, "BLEU\t" + whichone)





