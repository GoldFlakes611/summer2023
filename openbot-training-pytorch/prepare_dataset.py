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


def process_data(data_dir : str) -> None:
    '''
    Process data for training

    Args:
        data_dir (str): path to the dataset

    Returns:
        None
    '''
    # Initially filter and sort data. Only do this once/at any point after data is modified
    datasets = openbot.list_dirs(data_dir)

    log.info(f"Datasets: {len(datasets)}")

    # 1ms
    max_offset = 1e3
    frames = openbot.match_frame_ctrl_cmd(
        data_dir,
        datasets,
        max_offset,
        "train",
        redo_matching=True,
        remove_zeros=True,
    )

    image_count = len(frames)
    log.info("There are %d images" % (image_count))


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

    for data_dir in args.data_dir:
        process_data(data_dir)
