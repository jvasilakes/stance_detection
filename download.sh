DATA_DIR=data/

# ARC
mkdir -p $DATA_DIR/ARC/raw
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv $DATA_DIR/ARC/raw/bodies.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv $DATA_DIR/ARC/raw/stances_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv $DATA_DIR/ARC/raw/stances_test.csv

# FNC
mkdir -p $DATA_DIR/FNC/raw
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/train_bodies.csv $DATA_DIR/FNC/raw/bodies_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/train_stances.csv $DATA_DIR/FNC/raw/stances_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/test_bodies.csv $DATA_DIR/FNC/raw/bodies_test.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/competition_test_stances.csv $DATA_DIR/FNC/raw/stances_test.csv
cp $DATA_DIR/FNC/raw/bodies_train.csv $DATA_DIR/FNC/raw/bodies.csv
tail -n +2 $DATA_DIR/FNC/raw/bodies_test.csv >> $DATA_DIR/FNC/raw/bodies.csv

rm -rf coling2018_fake-news-challenge

# RumourEval 2019
wget -O $DATA_DIR/rumoureval.tar.bz2 https://figshare.com/ndownloader/files/16188500
tar xvf $DATA_DIR/rumoureval.tar.bz2 -C $DATA_DIR/

# RumourEval 2017
mkdir $DATA_DIR/rumoureval2017
wget -O $DATA_DIR/rumoureval2017/semeval2017-task8-dataset.tar.bz2 https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2
wget -O $DATA_DIR/rumoureval2017/semeval2017-task8-test-data.tar.bz2 http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2
tar xvf $DATA_DIR/rumoureval2017/semeval2017-task8-dataset.tar.bz2 -C $DATA_DIR/rumoureval2017/
tar xvf $DATA_DIR/rumoureval2017/semeval2017-task8-test-data.tar.bz2 -C $DATA_DIR/rumoureval2017/
# The data loader expects a topic directory before the actual tweets.
echo "mkdir $DATA_DIR/default_topic"
mkdir $DATA_DIR/rumoureval2017/semeval2017-task8-test-data/default_topic
mv $DATA_DIR/rumoureval2017/semeval2017-task8-test-data/* $DATA_DIR/rumoureval2017/semeval2017-task8-test-data/default_topic/
wget -O $DATA_DIR/rumoureval2017/test_taska.json http://alt.qcri.org/semeval2017/task8/data/uploads/subtaska.json

# AraStance
git clone https://github.com/Tariq60/arastance.git
mv arastance $DATA_DIR/
