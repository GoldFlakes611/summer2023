'''
Name: prepare_dataset.py
Description: This program is used to prepare the dataset for training
Date: 2023-08-25
Date Modified: 2023-08-25
'''
import openbot
from klogs import kLogger

TAG = "PREPARE"
log = kLogger(TAG)


if __name__ == "__main__":
    '''
    Examples:
        To process data in a directory:
            python prepare_dataset.py dataset/outside
        To view directory structure:
            tree --filelimit 10 dataset/outside
    '''
    import argparse
    
    argparser = argparse.ArgumentParser(description='Process data for training')
    argparser.add_argument('data_dir', nargs='+', type=str, help='Paths to the dataset')
    args = argparser.parse_args()

    frames = openbot.match_frame_ctrl_cmd(
        args.data_dir,
        max_offset=1e3,
        train_or_test="train",
        redo_matching=True,
        remove_zeros=True,
    )

    image_count = len(frames)
    log.info("There are totally %d images" % (image_count))
