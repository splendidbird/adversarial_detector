import requests
import json

class ctf(object):
    def __init__(self, url, attackid):
        self._url = url
        self._attackid = attackid
        # get my defenseid

    def getmydefense(self):
        payload = {
            'cmd' : 'getmydefense',
            'attackid' : self._attackid
        }
        r = requests.get(self._url, params=payload)
        print(r.url)
        json_data = r.json()
        print(json_data)
        if (json_data['result'] != 0):
            print('Result = %s Something wrong!!' % json_data['result'])
            exit(1)
        else:
            mydefenseid = json_data['defenseid']
            print('our defenseid = %s' % mydefenseid)
            return mydefenseid

    def getdefense(self):
        payload = {
            'cmd' : 'getdefenses',
            'attackid' : self._attackid
        }
        r = requests.get(self._url, params=payload)
        print(r.url)
        # get list of dict {classlabel:, defenseid: }
        json_data = r.json()
        # print(json_data[0]['defenseid'])
        defense_ids = [item['defenseid'] for item in json_data]
        for item in json_data:
            print('defenseid = %s' % item['defenseid'])
            print('classlable = %s' % item['classlabel'])
        return defense_ids

    
    def getpng(self, png_file):
        payload = {
            'cmd' : 'getteamlogo',
            'attackid' : self._attackid
        }
        r = requests.get(self._url, params=payload)
        print(r.url)
        print('%s saved' % png_file)
        open(png_file, 'wb').write(r.content)
        return r.content

    def postpng(self, inputpng, defenseid, epsilon):
        payload = {
            'cmd' : 'postattack',
            'attackid' : self._attackid,
            'defenseid' : defenseid,
            'epsilon' : epsilon
        }
        data = open(inputpng, 'rb').read()
        r = requests.post(self._url, params = payload, data = data)
        print(r.url)
        json_data = r.json()
        print(json_data)
        if (json_data['result'] != 0):
            print('Result = %s Something wrong!!' % json_data['result'])
            exit(1)
        else:
            classlabel = json_data['classlabel']
            attackresult = json_data['attackresult']
            print(attackresult)
            print('classlabel = %s' % classlabel)
            print('Description = %s' % json_data['description'])            
            return classlabel, attackresult

if __name__ == '__main__':
    url = 'http://35.230.92.142:8080'
    attackid = 'F98Z16'

    postget = ctf(url, attackid)
    # get our defense id and classlabel
    postget.getmydefense()

    # get opponent's denfense id
    postget.getdefense()

    # get image
    pngfile = '1.png'
    postget.getpng(pngfile)

    # post atttack
    defenseid = str(1)
    postget.postpng('1.png', defenseid, str(32))