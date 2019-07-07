



python export_graph.py --checkpoint_dir checkpoints/20190304-1210 \
                          --XtoY_model Good2Bad.pb \
                          --YtoX_model Bad2Good.pb \
                          --image_size 128



python build_data.py \
  --X_input_dir data/Good2Bad/TrainA \
  --Y_input_dir data/Good2Bad/TrainB \
  --X_output_file data/tfrecords/Good.tfrecords \
  --Y_output_file data/tfrecords/Bad.tfrecords 
  
  
python train.py  \
    --X data/tfrecords/Good.tfrecords \
    --Y data/tfrecords/Bad.tfrecords \ 
	--image_size 128
	
tensorboard --logdir checkpoints/20190311-1214

