import requests
import gzip
import glob
import os
import sys


def is_downloaded():
    # quick check for data download (not exhaustive)
    return bool(glob.glob("/data/*"))


def download_file(url, filename):
    r = requests.get(url)

    if r.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(r.content)
    else:
        print("err")


def get_url_filename(url):
    # may be done with urlparse
    return url.split('/')[-1]


def main():
    if is_downloaded():
        print("Data already downloaded (for re-download remove data folder)")
        sys.exit(0)

    sources = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]

    print("DOWNLOAD")
    for i in sources:
        pth = "/data/{}".format(get_url_filename(i))
        if not os.path.exists(pth):
            print("Downloading {} to {}".format(i, pth))
            download_file(i, pth)
        else:
            print("{} already downloaded".format(pth))

    print("UNPACK")
    for i in glob.glob("/data/*.gz"):
        pth = ''.join("/data/{}".format(os.path.basename(i)).split('.')[:-1])
        if not os.path.exists(pth):
            print("Unpacking {}".format(pth))
            with gzip.open(i) as f:
                with gzip.open(pth, 'wb') as out:
                    out.write(f.read())
        else:
            print("{} already unpacked".format(pth))


if __name__ == "__main__":
    main()
