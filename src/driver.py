import fnmatch as fn
import os
import re
from vectorModel import PositionDependentVectorModel

def matchingFiles(data_dirs, suffixes):
    globs = map(lambda x : '*.' + x, suffixes)
    matches = []  # list of files to read
    for data_dir in data_dirs:
        for root, dirnames, filenames in os.walk(data_dir):
            for glob in globs:
                for filename in fn.filter(filenames, glob):
                    matches.append(os.path.join(root, filename))
    return matches

def tokenize(fileName):
    allTokens = []
    with open(fileName) as data:
        allTokens += [token 
                      for token in re.split('\W+', data.read()) 
                      if len(token) > 0]
    return allTokens

if __name__ == '__main__':
    files = matchingFiles(['../data/linux'], ['c', 'h'])
    filesAndTokens = []
    for i,name in enumerate(files[:5000]):
        if i % 1000 == 0:
            print '%d files done' % i
        filesAndTokens.append((name,tokenize(name)))
#    print len(filesAndTokens)
#    print sum([len(tokens) for name,tokens in filesAndTokens])

    keywords = []
    with open('../key_words/c') as fp:
        for line in fp:
            keywords.append(line.strip().split()[0])
    print keywords
    model = PositionDependentVectorModel(keywords, winSize=100, wdim=48)
    model.train(filesAndTokens)
