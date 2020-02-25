import os
import tempfile
import random
import fire
import tensorflow as tf
from tfrecord import dataset_util
from tfrecord.utils import chunks


def create_tf_example(filepath):
    filename = os.path.basename(filepath)

    distance = float(os.path.splitext(filename)[0].split("_").pop())
    encoded_image_data = tf.io.gfile.GFile(filepath, "rb").read()

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/filename": dataset_util.bytes_feature(filename.encode()),
                "image/encoded": dataset_util.bytes_feature(encoded_image_data),
                "image/format": dataset_util.bytes_feature("jpeg".encode()),
                "image/distance": dataset_util.float_list_feature([distance]),
            }
        )
    )
    return tf_example


def create_tf_recrod(dst, filepaths):
    with tf.io.TFRecordWriter(dst) as writer:
        for filepath in filepaths:
            tf_example = create_tf_example(filepath)
            if not tf_example:
                continue
            writer.write(tf_example.SerializeToString())


def get_image_paths_chunks(src, directory, chunk_size=100):
    pattern = os.path.join(src, directory, "**", "*.jpg")
    image_paths = tf.io.gfile.glob(pattern)
    random.shuffle(image_paths)
    return chunks(image_paths, chunk_size)


def create_tf_record_shard(dst, directory, image_paths_chunks):
    tmp_dirpath = tempfile.mkdtemp()

    tmp_dsts = []

    for idx, chunk in enumerate(image_paths_chunks):
        prefix = "validation" if (idx + 1) % 6 == 0 else "train"
        record_filename = "{}-{}-{}.tfrecord".format(prefix, directory, idx)
        tmp_dst = os.path.join(tmp_dirpath, record_filename)
        create_tf_recrod(tmp_dst, chunk)
        tmp_dsts.append(tmp_dst)

    for record_path in tmp_dsts:
        record_filename = os.path.basename(record_path)
        copy_dst = os.path.join(dst, directory, record_filename)
        tf.io.gfile.copy(record_path, copy_dst, True)


def main(
    src="s3://pi-camera/forward-head-posture",
    dst="s3://tfrecord/forward-head-posture",
):
    src_dirs = sorted(tf.io.gfile.listdir(src))
    src_dirs = src_dirs[:-1]  # skip latest date

    dst_dirs = tf.io.gfile.listdir(dst)
    dirs = [x for x in src_dirs if x not in dst_dirs]
    print("target directory: ", dirs)

    for directory in dirs:
        print("start: ", directory)
        image_paths_chunks = get_image_paths_chunks(src, directory)
        create_tf_record_shard(dst, directory, image_paths_chunks)


if __name__ == "__main__":
    fire.Fire(main)
