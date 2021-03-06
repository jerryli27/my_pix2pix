"""
This file is for testing trained models by feeding them various types of input images.
"""
import os
import shutil

input_dirs = ["/mnt/06CAF857CAF84489/datasets/mnt/data_drive/home/ubuntu/pixiv_new_128_w_face_reshape_combined/test",
              "sketches_combined",
              "test_collected_colored_png",
              "test_collected_sketches_grayscale",
              "test_collected_colored_resized",
              "test_collected_sketches_resized",
              "test_collected_sketches_cropped",
              "test_collected_sketches_process_cropped",
              # "test_sketch_real_life",
              "/mnt/06CAF857CAF84489/datasets/shirobako_training_dataset/shirobako01pic_random100_128",
              "/mnt/06CAF857CAF84489/datasets/Demi-chan_wa_Kataritai/ed_resized",
              "/mnt/06CAF857CAF84489/datasets/Demi-chan_wa_Kataritai/ed_cleaned",
              "/mnt/1ACC040DCC03E1BD/pixiv_datasets/sketch_colored_pair_128_combined/test",
              "test_collected_sketches_cropped_old_sketch/sketch",
              "/home/jerryli27/PycharmProjects/my_pix2pix/result_page/pix2pix_w_hint_lab_wgan_larger_sketch_mix_cond_refactored_bn_train_sketch_test/outputs",
              "/home/jerryli27/PycharmProjects/my_pix2pix/result_page/pix2pix_w_hint_lab_wgan_larger_sketch_mix_cond_refactored_bn_train_sketch_mix_0_test/outputs",
              "/mnt/1ACC040DCC03E1BD/pixiv_datasets/sketch_convert_refactored_bn_train_sketch_mix_0_test/images",
              "test_collected_sketches_cropped_converted/images",
              ]

output_subdirs = ["test",
                  "test_sketches_combined",
                  "test_collected_colored",
                  "test_collected_sketches",
                  "test_collected_colored_resized",
                  "test_collected_sketches_resized",
                  "test_collected_sketches_cropped",
                  "test_collected_sketches_process_cropped",
                  # "test_sketch_real_life",
                  "test_shirobako",
                  "test_demichan_original",
                  "test_demichan_cleaned",
                  "test_sketch_colored_pair",
                  "test_collected_sketches_cropped_old_sketch",
                  "test_sketch_colored_pair_gen_sketch_mix_half",
                  "test_sketch_colored_pair_gen_sketch_mix_0",
                  "test_sketch_colored_pair_testset_sketch_convert",
                  "test_collected_sketches_cropped_converted",
                  ]

input_types = ["colored_combined",   # Modes: no hint, with hint, old sketch, new sketch
               "sketches_combined",  # Modes: no hint, input_sketch
               "colored_single",     # Modes: no hint, with hint, old sketch,
               "sketches_single",    # Modes: no hint, input_sketch
               "colored_single",     # Modes: no hint, with hint, old sketch,
               "sketches_single",    # Modes: no hint, input_sketch
               "sketches_single",    # Modes: no hint, input_sketch
               "sketches_single",    # Modes: no hint, input_sketch
               # "sketches_single",    # Modes: no hint, input_sketch
               "colored_single",     # Modes: no hint, with hint, old sketch,
               "colored_single",     # Modes: no hint, with hint, old sketch,
               "sketches_single",    # Modes: no hint, input_sketch
               "colored_combined",  # Modes: no hint, input_sketch
               "sketches_single",    # Modes: no hint, input_sketch
               "sketches_single",    # Modes: no hint, input_sketch
               "sketches_single",    # Modes: no hint, input_sketch
               "sketches_single",    # Modes: no hint, input_sketch
               "sketches_single",    # Modes: no hint, input_sketch
               ]

input_type_hint_modes = {"colored_combined": ["no_hint", "with_hint"],
                         "sketches_combined": ["no_hint", ],
                         "colored_single": ["no_hint", "with_hint"],
                         "sketches_single": ["no_hint", ],
                         }
input_type_sketch_modes = {"colored_combined": ["old_sketch", "input_sketch"],
                         "sketches_combined": ["input_sketch", ],
                         "colored_single": ["old_sketch",],
                         "sketches_single": ["input_sketch", ],
                         }

for checkpoint_subdir in [
                          "pixiv_2_drive_128_refactored_bn_noise_width_l1w_lrelu_grad_no_l1"]:
    # checkpoint_subdir = "pixiv_new_128_w_hint_lab_wgan_larger_sketch_mix_cond_refactored_bn_rerun"

    result_page_dir = "result_page/%s_test" % (checkpoint_subdir)
    if not os.path.isdir(result_page_dir):
        os.mkdir(result_page_dir)

    for i in range(len(input_dirs)):# range(len(input_dirs)):  # [6, 11, 13, 14, 15, 16]:
        for hint_mode in input_type_hint_modes[input_types[i]]:
            if hint_mode == "no_hint":
                hint_prob = -1
            elif hint_mode == "with_hint":
                hint_prob = 1
            else:
                raise AttributeError("Wrong hint mode %s." %(hint_mode))
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
                os.system("/usr/bin/python pix2pix_refactored_bn_noise_width_l1w_lrelu_grad.py --mode test --output_dir "
                          "%s "
                          "--max_epochs 20 --input_dir %s --which_direction AtoB "
                          "--display_freq=1000 --gray_input_a --batch_size 1 --lr 0.0008 --gpu_percentage 0.2 "
                          "--scale_size=143 --crop_size=128 --use_sketch_loss "
                          "--pretrained_sketch_net_path checkpoints/sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_train_sketch "
                          "--use_hint --lab_colorization --sketch_weight=10.0 --hint_prob=%f --gan_weight=5.0 "
                          "--checkpoint=checkpoints/%s "
                          "%s --mix_prob=%f" % (output_dir, input_dirs[i], hint_prob, checkpoint_subdir,
                                                single_input_string, mix_prob))