@ echo on
cd /d D:\PycharmProjects\CycleGan
call activate keras35

python build_data.py --X_input_dir data/Good2Bad/Train_Good  --Y_input_dir data/Good2Bad/Train_Bad --X_output_file data/tfrecords/Good.tfrecords --Y_output_file data/tfrecords/Bad.tfrecords  

python train.py --max_iteration 100 --save_ckpt_iteration 30 --batch_size 1 --image_size 256 --use_lsgan True --norm instance --lambda1 10 --lambda2 10 --learning_rate 0.0002 --beta1 0.5 --pool_size 50 --ngf 64 --X data/tfrecords/Good.tfrecords --Y data/tfrecords/Bad.tfrecords

python export_graph.py --save_model_dir pretrained --checkpoint_dir checkpoints --XtoY_model Good2Bad.pb --image_size 256

python inference.py --Test_input data/Test_input --Test_output data/Test_output --Model_dir pretrained  --image_size 256


pause







