# Homework 1

## Problem 1 - Intent Classification

<strong>Train</strong>

    python3 intent_cls_train.py \
        --train_data=<path_to_train.json> \
        --validation_data=<path_to_eval.json> \
        --checkpoint_directory=<where_to_save_checkpoint> \
        --checkpoint=<checkpoint_name> \
        --max_length=<max_length_of_input_sequence> \
        --batch_size=<batch_size> \
        --learning_rate=<learning_rate> \
        --epoch=<number_of_epoch>

<strong>Test</strong>

    python3 intent_cls_test.py \
        --test_data=<path_to_test.json> \
        --output_file=<path_to_save_prediction> \
        --checkpoint_directory=<where_the_checkpoints_are_saved> \
        --max_length=<max_length_of_input_sequence> \
        --batch_size=<batch_size>

## Problem 2 - Slot Tagging

<strong>Train</strong>

    python3 slot_tag_train.py \
        --train_data=<path_to_train.json> \
        --validation_data=<path_to_eval.json> \
        --checkpoint_directory=<where_to_save_checkpoint> \
        --checkpoint=<checkpoint_name> \
        --max_length=<max_length_of_input_sequence> \
        --batch_size=<batch_size> \
        --learning_rate=<learning_rate> \
        --epoch=<number_of_epoch>

<strong>Test</strong>

    python3 slot_tag_test.py \
        --test_data=<path_to_test.json> \
        --output_file=<path_to_save_prediction> \
        --checkpoint_directory=<where_the_checkpoints_are_saved> \
        --max_length=<max_length_of_input_sequence> \
        --batch_size=<batch_size>
