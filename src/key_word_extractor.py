#!/usr/bin/env python
# This file includes implementations for extracting key words from files.

from collections import Counter
import fnmatch as fn
import math
import os
import re
from utils import matchingFiles, tokenize

def FrequentWords(data_dirs, suffixes, max_key_words):
  """
  Returns a dictionary of min(max_key_words, percentile_key_words), giving key
  word with its count.
  """
  matches = matchingFiles(data_dirs, suffixes)

  token_count = Counter()
  files_done = 0
  for file_name in matches:
    tokens = tokenize(file_name)
    for token in tokens:
      if len(token) == 0:
        continue
      try:
        token_count[token] += 1
      except:
        token_count[token] = 1
    files_done += 1
    if (files_done % 5000 == 0):
      print("Completed parsing %d files ..." % files_done)

#  num_key_words = min(max_key_words,
#                      math.ceil(percentile_key_words * len(token_count)))
  return token_count.most_common(max_key_words)
