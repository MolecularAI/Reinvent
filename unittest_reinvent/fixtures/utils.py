import warnings
import os


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test

def count_empty_files(folder: str) -> int:
    empty_files_count = 0
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            if os.path.getsize(itempath) == 0:
                empty_files_count += 1
        # We already know that 'itempath' exists, so if it is not a file it must be a directory
        else:
            empty_files_count += count_empty_files(itempath)
    return empty_files_count
