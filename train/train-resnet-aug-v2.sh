python2 train.py \
--data_url='' \
--data_dir='./dataset' \
--summaries_dir='./summary/resnet-v2' \
--train_dir='./model/resnet-v2' \
--model_architecture='resnet' \
--batch_size=100 \
--learning_rate_start=0.0005 \
--learning_rate_decay=0.5 \
--epochs=20000 \
--restore_step_interval=1000 \
--stretch=0.2 \
--time_shift_ms=150 \
--background_frequency=0.9 \
--background_volume=0.15 \
--unknown_percentage=25.0 \
--dct_coefficient_count=40 \
--window_size_ms=25 \
--window_stride_ms=10 \
--weight_noise_stddev_start=0.001 \
--weight_noise_stddev_decay=0.2 \
--eval_step_interval=100 \
--testing_percentage=0 \
--validation_percentage=20 \
