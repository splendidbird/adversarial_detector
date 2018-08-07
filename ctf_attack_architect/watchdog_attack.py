from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time

from watchdog.observers import Observer
from watchdog.events import *
#from category.category import CategoryHelper

from NWS_Attack import *

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, nwsec):
        FileSystemEventHandler.__init__(self)
        self._nwsec = nwsec

    def on_moved(self, event):
        if event.is_directory:
            print("directory moved from {0} to {1}".format(event.src_path, event.dest_path))
        else:
            print("file moved from {0} to {1}".format(event.src_path, event.dest_path))

    def _attack_with_img_created(self, img_path_file):
        """ defense one image: xxx.png,
            write res to xxx.txt with two line(lable human_string),
            copy the src image file to output dir then delete it
        :param img_file:
        :return None:
        """
        img_file = img_path_file.split('/')[-1]
        img_path = '/'.join(img_path_file.split('/')[0:-1])
        if img_file.endswith('.png') and not img_file.startswith('._'):
            print(img_file)
            print(self._nwsec._attackid)
            # postpng deletes the file after attacking
            self._nwsec.postpng(img_file, img_path)

    def on_created(self, event):
        if event.is_directory:
            print("directory created:{0}".format(event.src_path))
        else:
            print("file created:{0}".format(event.src_path))
            self._attack_with_img_created(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            print("directory deleted:{0}".format(event.src_path))
        else:
            print("file deleted:{0}".format(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))


def main():

    with open("url_attackid.txt",'r') as f:
        [url, attackid] = f.read().splitlines()
    nwsec = NWS_Attack(url, attackid)
    # get our defense id and original classlabel
    nwsec.getmyid()
    # get attack mission list
    nwsec.getattacklist()

    observer = Observer()
    event_handler = FileEventHandler(nwsec)

    inputdir = "./adv_images/"

    observer.schedule(event_handler, inputdir, recursive=True)
    observer.start()
    print("attacking watchdog started...")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("\nattacking watchdog stoped!")

if __name__ == '__main__':
    main()
