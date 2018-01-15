python2 submit.py \
--data_url='' \
--data_dir='./dataset' \
--model_architecture='resnet' \
--start_checkpoint='./model/resnet-aug/best/resnet_9502.ckpt-14000' \
--LB_test_set_path='./testset/' \
--LB_test_batch_size=200 \
--submission_file_name='submission.csv'
