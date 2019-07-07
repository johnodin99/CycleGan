
python build_data.py \
  --X_input_dir data/Good2Bad/train_Good \
  --Y_input_dir data/Bad2Good/train_Bad \
  --X_output_file data/tfrecords/Good.tfrecords \
  --Y_output_file data/tfrecords/Bad.tfrecords