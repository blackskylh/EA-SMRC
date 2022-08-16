# path of the fine-tuned checkpoint
MODEL_PATH=$1
SPLIT=json
# input file that you would like to decode
INPUT_JSON=./DATA/unilm/unilm_splt_dev.json
dev_file=./DATA/tweetqa/dev.json
CUDA_VISIBLE_DEVICES=6 python3 ./decoder.py \
  --tokenizer_name ${MODEL_PATH} \
  --dev_file $dev_file \
  --model_type unilm --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 512 --max_tgt_length 32 --batch_size 64 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "." --do_rule 
