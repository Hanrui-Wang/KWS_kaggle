python2 train.py \
--data_url='' \
--data_dir='./dataset' \
--summaries_dir='./summary/ml-bigru-aug-noise-nomfcc' \
--train_dir='./model/ml-bigru-aug-noise-nomfcc' \
--model_architecture='bigru' \
--batch_size=100 \
--learning_rate_start=0.0005 \
--learning_rate_decay=0.5 \
--epochs=20000 \
--restore_step_interval=800 \
--stretch=0.3 \
--time_shift_ms=150 \
--background_frequency=0.9 \
--background_volume=0.18 \
--unknown_percentage=25.0 \
--dct_coefficient_count=40 \
--window_size_ms=25 \
--window_stride_ms=10 \
--weight_noise_stddev_start=0.001 \
--weight_noise_stddev_decay=0.2 \
--eval_step_interval=100 \
--is_use_mfcc='' \
--testing_percentage=0 \
--validation_percentage=20 \
