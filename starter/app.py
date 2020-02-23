import time
import fire
import requests


def main(sleep_time=0.01):
    time.sleep(sleep_time)
    return requests.get("https://github.com")


if __name__ == "__main__":
    fire.Fire(main)
