DATA_DIR=data/

mkdir $DATA_DIR/ARC
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv $DATA_DIR/ARC/bodies.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv $DATA_DIR/ARC/stances_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv $DATA_DIR/ARC/stances_test.csv

mkdir $DATA_DIR/FNC
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/train_bodies.csv $DATA_DIR/FNC/bodies_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/train_stances.csv $DATA_DIR/FNC/stances_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/test_bodies.csv $DATA_DIR/FNC/bodies_test.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/competition_test_stances.csv $DATA_DIR/FNC/stances_test.csv
cat $DATA_DIR/FNC/bodies_train.csv $DATA_DIR/FNC/bodies_test.csv > $DATA_DIR/FNC/bodies.csv

rm -rf coling2018_fake-news-challenge
