"""
This file is for testing trained models by feeding them various types of input images.
"""
import os
import shutil

input_dirs = ["palette_sanity_check.tsv",
              "/home/jerryli27/PycharmProjects/mmcq.py/pixiv_new_128_w_face_reshape_combined_palette_test.tsv",
              "/home/jerryli27/PycharmProjects/mmcq.py/sketch_colored_pair_palette_test.tsv",
              # "/home/jerryli27/PycharmProjects/mmcq.py/sketch_colored_pair_palette_train.tsv",
              ]

output_subdirs = ["test_sanity_check",
                  "test",
                  "test_sketch_colored_test",
                  # "test_sketch_colored_train",
                  ]

input_types = ["combined_with_palette",
               "combined_with_palette",
               "combined_with_palette",
               # "combined_with_palette",
               ]

input_type_palette_modes = {"combined_with_palette": ["with_palette"],
                            }
input_type_sketch_modes = {"combined_with_palette": ["old_sketch", "input_sketch"],
                         }


checkpoint_subdir = "pixiv_new_128_palette_ver2_no_hint"

result_page_dir = "result_page/%s_test" % (checkpoint_subdir)
if not os.path.isdir(result_page_dir):
    os.mkdir(result_page_dir)

for i in range(len(input_dirs)):  # [6, 11]:
    for hint_mode in input_type_palette_modes[input_types[i]]:
        if hint_mode == "with_palette":
            pass
        # Haven't implemented no palette mode.
        else:
            raise AttributeError("Wrong palette mode %s." %(hint_mode))
        for sketch_mode in input_type_sketch_modes[input_types[i]]:

            if sketch_mode == "old_sketch":
                mix_prob = 1
            elif sketch_mode == "input_sketch":
                mix_prob = -1
            else:
                raise AttributeError("Wrong sketch mode %s." % (sketch_mode))
            current_mode = sketch_mode + "_" + hint_mode
            output_dir = "result_page/%s_test/%s_%s" %(checkpoint_subdir, output_subdirs[i], current_mode)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            single_input_string = "--single_input" if input_types[i].endswith("single") else ""
            os.system("/usr/bin/python pix2pix_w_hint_lab_wgan_larger_sketch_mix_cond_refactored_dilation_deeper_palette_ver2.py --mode test --output_dir "
                      "%s "
                      "--max_epochs 20 --input_dir %s --which_direction AtoB "
                      "--display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.25 "
                      "--scale_size=143 --crop_size=128 --use_sketch_loss "
                      "--pretrained_sketch_net_path checkpoints/sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_train_sketch "
                      "--lab_colorization --sketch_weight=10.0 "
                      "--checkpoint=checkpoints/%s "
                      "%s --mix_prob=%f" % (output_dir, input_dirs[i], checkpoint_subdir,
                                            single_input_string, mix_prob))