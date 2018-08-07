from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import os
import glob
import logging
import numpy as np
import pandas as pd
from shutil import copyfile
import csv
import datetime
from time import strftime
from printcsv import *



class NWS_Attack(object):

    """ NWS_Attack class is responsible for entire attacking procedures
        Step 1. Acquire Attacking info (orig_images, defensers and target labels)
        Step 2. Send info to appointed folders for adversarial processing
        Step 3. Acquir adv samples from appointed folders
        Step 4. Send out attacks with adv samples
        Step 5. Acquire, organize, display and store attacking result feedback.
    """


    def __init__(self, url, attackid,
                 datapath="dataset/images/",
                 datafile="target_class.csv",
                 attackpath="send_attack/",
                 attackalgorithm=["momentum/","ToshiK/"]):

        self._url = url
        self._attackid = attackid

        self._datapath = datapath
        self._datafile = datafile

        self._datafile = datafile
        self._attackpath = attackpath

        self._mydefenseid = None
        self._origlabel = None
        self._attacklist = None

        self._attackalgorithm = attackalgorithm


    def getmyid(self):
        # To acquire our team's defense id and original label of assigned image

        payload = {
            'cmd' : 'getmydefense',
            'attackid' : self._attackid
        }
        r = requests.get(self._url, params=payload)
        print("Sent Request: getmydefense")
        print(r.url)
        json_data = r.json()
        print("Feedback Acquired:")
        print(json_data)
        if (json_data['result'] != 0):
            print('Error running getmyid()! ErrorCode(Result) = %s ' % json_data['result'])
            exit(1)
        else:
            self._mydefenseid = json_data['defenseid']
            self._origlabel = json_data['classlabel']
            print('Acquired my defenseid = %s' % self._mydefenseid)
            print('Acquired Original Label = %s' % self._origlabel)
            return self._mydefenseid, self._origlabel


    def getattacklist(self):
        # return attack list [(defenser1, target1),(defenser2, target2)...]

        payload = {
            'cmd' : 'getdefenses',
            'attackid' : self._attackid
        }
        r = requests.get(self._url, params=payload)
        print("Sent Request: getdefenses")
        print(r.url)
        # get list of dict {classlabel:, defenseid: }
        json_data = r.json()
        # print(json_data[0]['defenseid'])
        self._attacklist = [(item['defenseid'], item['classlabel']) for item in json_data]
        print("Acquired attack list:")
        print(self._attacklist)
        # attention - numbers not strings
        return self._attacklist


    def getpng(self):
        # acquire orig image and duplicate for each defenser
        # save into ./dataset/images folder

        payload = {
            'cmd' : 'getteamlogo',
            'attackid' : self._attackid
        }
        r = requests.get(self._url, params=payload)
        print(r.url)
        print('orig.png saved')
        open(os.path.join(self._datapath, "orig.png"), 'wb').write(r.content)
        return r.content


    def loadattack(self):
        # 1. Acquire & Organize defensers' ids and each of their targeted class
        # 2. Acquire & Organize the downloaded original images & duplicate for
        # each of the defenser
        # 3. Prepare "target_class.csv" file

        self.getattacklist()
        self.getpng()
        # original png
        opng = os.path.join(self._datapath, "orig.png")
        img_col = []
        for (defenser, _) in self._attacklist:
            #duplicated png filename
            img_col.append("orig_"+str(defenser)+".png")
            dpng= os.path.join(self._datapath, "orig_" + str(defenser) + ".png")
            copyfile(opng,dpng)
        os.remove(opng)
        print("orig images for all defensers generated in "+ self._datapath)

        attack_df = pd.DataFrame(np.array(self._attacklist), columns=['Defenser', 'TargetLabel'])
        attack_df["ImgFile"] = img_col
        # save to csv file "target_class.csv"
        csvfile = os.path.join(self._datapath, self._datafile)
        attack_df[["ImgFile", "TargetLabel"]].to_csv(csvfile, index=False, header=False)
        print("target_class.csv saved to " + self._datapath)
        # printout target_class.csv to confirm
        printcsv(csvfile)


    def postpng(self, imagename, path="./adv_images/"):
        # single adv image post for attack
        # feedback recorded in target_result.csv
        # adv image deleted

        if self._mydefenseid == None:
            print("Defense ID non-exists! Running .getmyid() ...")
            self.getmyid()

        defenserid = int(imagename[-5])
        epsilon = imagename.split('_')[-2]
        orig_label = self._origlabel
        target_label = str([x for (_,x) in self._attacklist if _==defenserid][0])

        payload = {
            'cmd' : 'postattack',
            'attackid' : self._attackid,
            'defenseid' : str(defenserid),
            'epsilon' : epsilon
        }
        data = open(os.path.join(path, imagename), 'rb').read()
        r = requests.post(self._url, params = payload, data = data)
        print(r.url)
        json_data = r.json()
        print(json_data)
        if (json_data['result'] != 0):
            print('Error running postpng()! ErrorCode(Result) = %s' % json_data['result'])
            exit(1)
        else:
            classlabel = json_data['classlabel']
            attackresult = json_data['attackresult']
            recordtime = datetime.datetime.now().strftime("%H:%M:%S")
            record = [recordtime,
                      defenserid,
                      orig_label,
                      target_label,
                      classlabel,
                      attackresult[:3],
                      epsilon,
                      '_'.join(imagename.split('_')[:-2])]

            with open('target_result.csv','a') as newRecord:
                newRecordWriter = csv.writer(newRecord)
                newRecordWriter.writerow(record)

            # delete image from adv_images folder
            os.remove(os.path.join(path, imagename))
            print(imagename + " deleted from " + path)
            return record

    def attacking(self, adv_path="./adv_images/"):
        # launch attacking pic by pic in the adv_path
        adv_images = [f for f in os.listdir(adv_path) if f.endswith(".png") and not f.startswith("._")]
        for imagename in adv_images:
            self.postpng(imagename)



# Example Demo: run by running "python NWS_Attack.py" directly

if __name__ == '__main__':

    # public
    url = 'http://35.230.92.142:8080'
    attackid = 'F98Z16'

    # local debugging
    url = 'http://127.0.0.1:8866'
    attackid = '12345'

    nwsec = NWS_Attack(url, attackid)
    # get our defense id and original classlabel
    nwsec.getmyid()

    # get attack mission list
    nwsec.getattacklist()

    # get orig image and load in attack list
    # prepared and saved properly
    nwsec.loadattack()

    # attack, record feedback result and delete used images
    nwsec.attacking()














# end of code
