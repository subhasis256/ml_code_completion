Instructions on how to run the current source code:

First, you need to extract keywords for the project and language(s) that you
are interested in. For example, if you are interested in linux project and all C
source files, i.e., files with .c or .h endings, you should do:

`./extract_key_words -p linux -l c,h -n 2000`

This will extract 2000 top most frequent tokens from the `linux` project and put
that in a file `../key_words/linux` file.

Next, you need to train and test using the `driver.py` file. It can be run in
three modes: training, testing or prediction. A summary of all options can be
obtained by doing:

`python driver.py -h`

For training, assuming the keyword extraction is already done, the `driver.py`
can be invoked thus:

`python driver.py train --ckpt <ckpt_prefix> --zdim 1024 -p linux -l c,h --lr 0.05 --reg 0.01`

This will output training statistics and will intermittently checkpoint the
current model to a pickle file (by default every 100000 batches).

For testing, you should do:

`python driver.py test --restore <saved_param_file> --zdim 1024 -p linux -l c,h`

This will output several testing metrics.

For prediction, you should do:

`python driver.py predict --restore <saved_param_file> --zdim 1024 -p linux -l c,h`

This will take random files from the linux directory and do
character-by-character prediction on those files. Subsequently it will write an
annotated html file into the current directory.
