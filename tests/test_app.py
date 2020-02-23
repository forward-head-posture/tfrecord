import time
import responses
from tfrecord.app import main


@responses.activate
def test_main(mocker):
    mocker.patch("time.sleep")
    responses.add(responses.GET, "https://github.com", status=200, json={})
    sleep_time = 0.1
    response = main(sleep_time)
    time.sleep.assert_called_with(sleep_time)  # pylint: disable=no-member
    assert response.status_code == 200
