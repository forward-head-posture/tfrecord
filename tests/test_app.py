# pylint: disable=line-too-long,no-member

import io
import pytest
import tensorflow as tf
import tfrecord
from tfrecord.app import (
    main,
    get_image_paths_chunks,
    create_tf_record_shard,
    create_tf_recrod,
    create_tf_example,
)


def test_main(mocker):
    directory = "2020-02-24"
    mocker.patch("tensorflow.io.gfile.listdir").return_value = [directory]
    image_paths_chunks = [
        [
            "s3://pi-camera/forward-head-posture/2020-02-24/00/1582505488184000000_1582505488196884480_pi-camera-3_0.0334572092477628.jpg"
        ],
        [
            "s3://pi-camera/forward-head-posture/2020-02-24/00/1582504169424000000_1582504169467478784_pi-camera-1_0.008780005006323754.jpg"
        ],
    ]
    mocker.patch("tensorflow.io.gfile.glob").return_value = image_paths_chunks
    mocker.patch("tfrecord.app.create_tf_record_shard")
    main()
    assert tfrecord.app.create_tf_record_shard.call_args[0][1] == directory


def test_get_image_paths_chunks(mocker):
    mocker.patch("random.shuffle")
    mocker.patch("tensorflow.io.gfile.glob").return_value = [
        "s3://pi-camera-3_0.0334572092477628.jpg",
        "s3://pi-camera-0_0.00456208523369156.jpg",
    ]
    src = "s3://pi-camera/forward-head-posture"
    directory = "2020-02-24"
    chunk_size = 1
    image_paths_chunks = list(
        get_image_paths_chunks(src, directory, chunk_size)
    )
    expected = [
        ["s3://pi-camera-3_0.0334572092477628.jpg"],
        ["s3://pi-camera-0_0.00456208523369156.jpg"],
    ]
    assert image_paths_chunks == expected


def test_create_tf_record_shard(mocker):
    mocker.patch("tensorflow.io.gfile.copy")
    mocker.patch("tensorflow.io.gfile.GFile").return_value = io.BytesIO(
        b"test-bytes"
    )
    dst = "s3://test/forward-head-posture"
    directory = "2020-02-24"
    image_paths_chunks = [
        [
            "s3://pi-camera/forward-head-posture/2020-02-24/00/1582505488184000000_1582505488196884480_pi-camera-3_0.0334572092477628.jpg",
            "s3://pi-camera/forward-head-posture/2020-02-24/00/1582503996107000000_1582503996153114368_pi-camera-0_0.00456208523369156.jpg",
        ],
        [
            "s3://pi-camera/forward-head-posture/2020-02-24/00/1582504169424000000_1582504169467478784_pi-camera-1_0.008780005006323754.jpg",
            "s3://pi-camera/forward-head-posture/2020-02-24/01/1582506958037000000_1582506958073802752_pi-camera-0_0.04352723969174874.jpg",
        ],
        [
            "s3://pi-camera/forward-head-posture/2020-02-24/01/1582507198013000000_1582507198005175040_pi-camera-3_0.003301300849721578.jpg"
        ],
    ]
    create_tf_record_shard(dst, directory, image_paths_chunks)
    args_list = tf.io.gfile.copy.call_args_list

    for i in range(len(image_paths_chunks)):
        assert args_list[i][0][
            1
        ] == "s3://test/forward-head-posture/2020-02-24/2020-02-24-{}.tfrecord".format(
            i
        )


def test_create_tf_example(mocker):
    mocker.patch("tensorflow.io.gfile.GFile").return_value = io.BytesIO(
        b"test-bytes"
    )
    distance = 0.0032342343244
    mocker.patch(
        "tfrecord.dataset_util.float_list_feature"
    ).return_value = tfrecord.dataset_util.float_list_feature([distance])
    filepath = "s3://pi-camera/forward-head-posture/2020-02-24/00/1582505488184000000_1582505488196884480_pi-camera-3_{}.jpg".format(
        distance
    )
    create_tf_example(filepath)
    tfrecord.dataset_util.float_list_feature.assert_called_with([distance])


TFRECORD_SAVE_PATH = "/tmp/train-01.tfrecord"


@pytest.mark.order1
def test_create_tf_record(mocker):
    mocker.patch("tensorflow.io.gfile.GFile").return_value = io.BytesIO(
        b"test-bytes"
    )
    filepaths = [
        "s3://pi-camera/forward-head-posture/2020-02-24/00/1582505488184000000_1582505488196884480_pi-camera-3_0.0334572092477628.jpg",
        "s3://pi-camera/forward-head-posture/2020-02-24/00/1582503996107000000_1582503996153114368_pi-camera-0_0.00456208523369156.jpg",
    ]
    create_tf_recrod(TFRECORD_SAVE_PATH, filepaths)


@pytest.mark.order2
def test_read_tfrecord():
    image_feature_description = {
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature((), tf.string),
        "image/format": tf.io.FixedLenFeature((), tf.string),
        "image/distance": tf.io.FixedLenFeature((), tf.float32),
    }
    raw_image_dataset = tf.data.TFRecordDataset(TFRECORD_SAVE_PATH)

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(
            example_proto, image_feature_description
        )

    raw_image_dataset.map(_parse_image_function)
