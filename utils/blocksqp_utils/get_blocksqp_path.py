from pathlib import Path


def get_path():
    """Get the path to the local BlockSQP installation from the blocksqp_path.txt file.
    """
    # get the absolute path of the txt file
    curr_path = Path(__file__).parent
    ref_file = open(str(curr_path) + "/../../blocksqp_path.txt")

    # first line is only a comment
    ref_file.readline()
    # remove \n character...
    path = str(ref_file.readline())[:-1]
    ref_file.close()

    return path

