# TODO: create shell script for Problem 3

# download model weights
mkdir -p storage

curl -L -o FVrnnVGG_RNN.pkl "https://github.com/dlcv-spring-2019/hw4-YiJingLin/releases/download/v1.0/FVrnnVGG_RNN.pkl"
curl -L -o FVrnnVGG_classifier.pkl "https://github.com/dlcv-spring-2019/hw4-YiJingLin/releases/download/v1.0/FVrnnVGG_classifier.pkl"

mv FVrnnVGG_RNN.pkl storage/FVrnnVGG_RNN.pkl
mv FVrnnVGG_classifier.pkl storage/FVrnnVGG_classifier.pkl

# main function
# python3 p3_eval.py ./hw4_data/FullLengthVideos/videos/valid ./output
python3 p3_eval.py $1 $2