#!/usr/bin/env python
# This file includes implementations for extracting key words from files.

from collections import Counter
import fnmatch as fn
import math
import os
import re

def FrequentWords(data_dirs, suffixes, max_key_words, percentile_key_words):
  """
  Returns a dictionary of min(max_key_words, percentile_key_words), giving key
  word with its count.
  """
  globs = map(lambda x : '*.' + x, suffixes)

  matches = []  # list of files to read
  for data_dir in data_dirs:
    for root, dirnames, filenames in os.walk(data_dir):
      for glob in globs:
        for filename in fn.filter(filenames, glob):
          matches.append(os.path.join(root, filename))
  num_files = len(matches)

  token_count = Counter()
  files_done = 0
  for file_name in matches:
    with open(file_name) as data:
      for line in data:
        tokens = re.split('\W+', line)
        for token in tokens:
          if len(token) == 0:
            continue
          try:
            token_count[token] += 1
          except:
            token_count[token] = 0
    files_done += 1
    if (files_done % 5000 == 0):
      print("Completed parsing %d files ...", files_done)

  num_key_words = min(max_key_words,
                      math.ceil(percentile_key_words * len(token_count)))
  return token_count.most_common(num_key_words)
