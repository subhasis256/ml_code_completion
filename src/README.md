Instructions on how to run the current source code:

The file to run is the `driver.py` file. It can be run in three modes: training,
testing or prediction. A summary of all options can be obtained by doing:

`python driver.py -h`

For training, assuming you already have downloaded the data, the `driver.py`
will train on the linux kernel source. You can do this by

`python driver.py train <TRAIN_OPTIONS>`

This will output training statistics and will intermittently checkpoint the
current model to a pickle file (by default every 100000 batches).

For testing, you should do:

`python driver.py test <TEST_OPTIONS>`

This will output several testing metrics.

For prediction, you should do:

`python driver.py predict <PREDICT_OPTIONS>`

This will take random files from the linux directory and do
character-by-character prediction on those files. Subsequently it will write an
annotated html file into the current directory.
