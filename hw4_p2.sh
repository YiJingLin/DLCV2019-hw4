# TODO: create shell script for Problem 2

# download model weights
mkdir -p storage

curl -L -o MFrnnVGG_RNN.pkl "https://github.com/dlcv-spring-2019/hw4-YiJingLin/releases/download/v1.0/MFrnnVGG_RNN.pkl"
curl -L -o MFrnnVGG_classifier.pkl "https://github.com/dlcv-spring-2019/hw4-YiJingLin/releases/download/v1.0/MFrnnVGG_classifier.pkl"

mv MFrnnVGG_RNN.pkl storage/MFrnnVGG_RNN.pkl
mv MFrnnVGG_classifier.pkl storage/MFrnnVGG_classifier.pkl

# main function
# python3 p2_eval.py ./hw4_data/TrimmedVideos/video/valid ./hw4_data/TrimmedVideos/label/gt_valid.csv ./output
python3 p2_eval.py $1 $2 $3