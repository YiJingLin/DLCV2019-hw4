# TODO: create shell script for Problem 1

# download model weights
mkdir -p storage
curl -L -o MFVGG_classifier.pkl "https://github.com/dlcv-spring-2019/hw4-YiJingLin/releases/download/v1.0/MFVGG_classifier.pkl"
mv MFVGG_classifier.pkl storage/MFVGG_classifier.pkl


# main function
# python3 p1_eval.py ./hw4_data/TrimmedVideos/video/valid ./hw4_data/TrimmedVideos/label/gt_valid.csv ./output
python3 p1_eval.py $1 $2 $3