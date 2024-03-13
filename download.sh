DATA_DIR=data/

# ARC
mkdir -p $DATA_DIR/ARC/raw
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv $DATA_DIR/ARC/raw/bodies.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv $DATA_DIR/ARC/raw/stances_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv $DATA_DIR/ARC/raw/stances_test.csv
python scripts/preprocess_dataset.py arc $DATA_DIR/ARC/raw $DATA_DIR/ARC/preprocessed > tmp.txt
mv tmp.txt $DATA_DIR/ARC/preprocessed_stats.txt

rm -rf coling2018_fake-news-challenge

# RumourEval 2019
wget -O $DATA_DIR/rumoureval.tar.bz2 https://figshare.com/ndownloader/files/16188500
tar xvf $DATA_DIR/rumoureval.tar.bz2 -C $DATA_DIR/
rm $DATA_DIR/rumoureval.tar.bz2
unzip $DATA_DIR/rumoureval2019/rumoureval-2019-training-data.zip -d $DATA_DIR/rumoureval2019
unzip $DATA_DIR/rumoureval2019/rumoureval-2019-test-data.zip -d $DATA_DIR/rumoureval2019
python scripts/preprocess_dataset.py rumoureval $DATA_DIR/rumoureval2019 $DATA_DIR/rumoureval2019/preprocessed/twitter -k version 2019 -k load_reddit False > tmp.txt
mv tmp.txt $DATA_DIR/rumoureval2019/preprocessed_stats_twitter.txt
python scripts/preprocess_dataset.py rumoureval $DATA_DIR/rumoureval2019 $DATA_DIR/rumoureval2019/preprocessed/twitter_reddit -k version 2019 -k load_reddit True > tmp.txt
mv tmp.txt $DATA_DIR/rumoureval2019/preprocessed_stats_twitter_reddit.txt

# RumourEval 2017
mkdir $DATA_DIR/rumoureval2017
wget -O $DATA_DIR/rumoureval2017/semeval2017-task8-dataset.tar.bz2 https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2
wget -O $DATA_DIR/rumoureval2017/semeval2017-task8-test-data.tar.bz2 http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2
tar xvf $DATA_DIR/rumoureval2017/semeval2017-task8-dataset.tar.bz2 -C $DATA_DIR/rumoureval2017/
tar xvf $DATA_DIR/rumoureval2017/semeval2017-task8-test-data.tar.bz2 -C $DATA_DIR/rumoureval2017/
# The data loader expects a topic directory before the actual tweets.
mkdir $DATA_DIR/rumoureval2017/semeval2017-task8-test-data/default_topic
mv $DATA_DIR/rumoureval2017/semeval2017-task8-test-data/* $DATA_DIR/rumoureval2017/semeval2017-task8-test-data/default_topic/
wget -O $DATA_DIR/rumoureval2017/test_taska.json http://alt.qcri.org/semeval2017/task8/data/uploads/subtaska.json
python scripts/preprocess_dataset.py rumoureval $DATA_DIR/rumoureval2017 $DATA_DIR/rumoureval2017/preprocessed -k version 2017 > tmp.txt
mv tmp.txt $DATA_DIR/rumoureval2017/preprocessed_stats.txt

# AraStance
git clone https://github.com/Tariq60/arastance.git
mv arastance $DATA_DIR/
python scripts/preprocess_dataset.py arastance $DATA_DIR/arastance/data $DATA_DIR/arastance/preprocessed > tmp.txt
mv tmp.txt $DATA_DIR/arastance/preprocessed_stats.txt

# Russian
git clone https://github.com/lozhn/rustance.git
mv rustance $DATA_DIR/
python scripts/preprocess_dataset.py rustance $DATA_DIR/rustance/Dataset $DATA_DIR/rustance/preprocessed > tmp.txt
mv tmp.txt $DATA_DIR/rustance/preprocessed_stats.txt
