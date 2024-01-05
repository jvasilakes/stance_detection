DATA_DIR=data/

mkdir -p $DATA_DIR/ARC/raw
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv $DATA_DIR/ARC/raw/bodies.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv $DATA_DIR/ARC/raw/stances_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv $DATA_DIR/ARC/raw/stances_test.csv

mkdir -p $DATA_DIR/FNC/raw
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/train_bodies.csv $DATA_DIR/FNC/raw/bodies_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/train_stances.csv $DATA_DIR/FNC/raw/stances_train.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/test_bodies.csv $DATA_DIR/FNC/raw/bodies_test.csv
mv coling2018_fake-news-challenge/data/fnc-1/corpora/FNC/competition_test_stances.csv $DATA_DIR/FNC/raw/stances_test.csv
cp $DATA_DIR/FNC/raw/bodies_train.csv $DATA_DIR/FNC/raw/bodies.csv
tail -n +2 $DATA_DIR/FNC/raw/bodies_test.csv >> $DATA_DIR/FNC/raw/bodies.csv

rm -rf coling2018_fake-news-challenge
