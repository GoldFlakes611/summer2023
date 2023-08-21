import openbot


def process_data(data_dir):
    # Initially filter and sort data. Only do this once/at any point after data is modified
    datasets = openbot.list_dirs(data_dir)

    print("Datasets: ", len(datasets))

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
    print("There are %d images" % (image_count))


if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser(description='Process data for training')
    argparser.add_argument('data_dir', nargs='+', type=str, help='Paths to the dataset')
    args = argparser.parse_args()

    for data_dir in args.data_dir:
        process_data(data_dir)
