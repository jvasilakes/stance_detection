DATA_DIR=data/

mkdir $DATA_DIR/ARC
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv $DATA_DIR/ARC
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv $DATA_DIR/ARC
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv $DATA_DIR/ARC
rm -rf coling2018_fake-news-challenge
