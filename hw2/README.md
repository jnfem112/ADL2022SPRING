# Homework 2

<strong>Train</strong>

    TOKENIZERS_PARALLELISM=false python3 train_1.py \
        --context_file=<path_to_context.json> \
        --train_data=<path_to_train.json> \
        --validation_data=<path_to_eval.json> \
        --base_model=<whether_to_use_BERT> \
        --pretrained=<whether_to_use_pretrained_model> \
        --plot=<whether_to_plot_the_learning_curve>

    TOKENIZERS_PARALLELISM=false python3 train_2.py \
        --context_file=<path_to_context.json> \
        --train_data=<path_to_train.json> \
        --validation_data=<path_to_eval.json> \
        --base_model=<whether_to_use_BERT> \
        --pretrained=<whether_to_use_pretrained_model> \
        --plot=<whether_to_plot_the_learning_curve>

    TOKENIZERS_PARALLELISM=false python3 bonux_intent_cls_train.py \
        --context_file=<path_to_context.json> \
        --train_data=<path_to_train.json> \
        --validation_data=<path_to_eval.json> \
        --base_model=<whether_to_use_BERT> \
        --pretrained=<whether_to_use_pretrained_model> \
        --plot=<whether_to_plot_the_learning_curve>

    TOKENIZERS_PARALLELISM=false python3 bonux_slot_tag_train.py \
        --context_file=<path_to_context.json> \
        --train_data=<path_to_train.json> \
        --validation_data=<path_to_eval.json> \
        --base_model=<whether_to_use_BERT> \
        --pretrained=<whether_to_use_pretrained_model> \
        --plot=<whether_to_plot_the_learning_curve>

<strong>Test</strong>

    TOKENIZERS_PARALLELISM=false python3 test.py \
        --context_file=<path_to_context.json> \
        --test_data=<path_to_test.json> \
        --base_model=<whether_to_use_BERT> \
        --output_file=<path_to_save_prediction>

    TOKENIZERS_PARALLELISM=false python3 bonux_intent_cls_test.py \
        --context_file=<path_to_context.json> \
        --test_data=<path_to_test.json> \
        --base_model=<whether_to_use_BERT> \
        --output_file=<path_to_save_prediction>

    TOKENIZERS_PARALLELISM=false python3 bonux_slot_tag_test.py \
        --context_file=<path_to_context.json> \
        --test_data=<path_to_test.json> \
        --base_model=<whether_to_use_BERT> \
        --output_file=<path_to_save_prediction>