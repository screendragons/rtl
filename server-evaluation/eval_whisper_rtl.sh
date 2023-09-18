# activating environment
. env/bin/activate

VERSION=v3
REFERENCE_FOLDER=/home2/rai_asr/datasets/jasmin-eval
JASMIN_DATASET_FOLDER=/home2/rai_asr/datasets/jasmin/Data/data/audio/wav
META_FOLDER=/home2/rai_asr/datasets/jasmin/Data/data/meta/text
WHISPER_MODEL=rtl-whisper-small

export PATH=/home2/rai_asr/utils/ffmpeg-git-20220910-amd64-static:$PATH

# comp-q nl

python eval.py $WHISPER_MODEL \
			$JASMIN_DATASET_FOLDER  \
			$REFERENCE_FOLDER  \
			$META_FOLDER \
			q nl ./results $VERSION

# comp-q vl

python eval.py $WHISPER_MODEL \
			$JASMIN_DATASET_FOLDER  \
			$REFERENCE_FOLDER \
			$META_FOLDER \
			q vl ./results $VERSION

# comp-p nl

python eval.py $WHISPER_MODEL \
			$REFERENCE_FOLDER \
			$REFERENCE_FOLDER  \
			$META_FOLDER \
			p nl ./results $VERSION

# comp-p vl

python eval.py $WHISPER_MODEL \
			$REFERENCE_FOLDER  \
			$REFERENCE_FOLDER  \
			$META_FOLDER \
			p vl ./results $VERSION

# concatenating results
head -1 results/eval-$WHISPER_MODEL-comp-p-nl-results-$VERSION.txt > results/eval-$WHISPER_MODEL-results-$VERSION.txt
tail -q -n +2 results/eval-$WHISPER_MODEL-comp-?-?l-results-$VERSION.txt >> results/eval-$WHISPER_MODEL-results-$VERSION.txt

# deactivating
deactivate
