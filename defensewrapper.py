#coding=utf-8
from BaseHTTPServer import HTTPServer,BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import threading
import cgi
import json
import datetime
import urllib
from urlparse import parse_qs
import numpy as np
from PIL import Image
import math
import cStringIO
import time
import os.path
from threading import Thread, Lock

mutex = Lock()
teamlabels = [
        {"attackid":"12345","defenseid":1,"classlabel":34},
        {"attackid":"54321","defenseid":2,"classlabel":260},
        {"attackid":"abcde","defenseid":3,"classlabel":851},
        {"attackid":"aaa67","defenseid":4,"classlabel":74},
        {"attackid":"zz610","defenseid":5,"classlabel":309}
    ] #you can change to other values when you test
inputpath = './input/'
outputpath = './output/'
imageArr = {}
imageMaxArr = {}
imageMinArr = {}
attackSequence = 0

def loadteamlogos():
    for i in range(1,6):        
        image = np.array(
            Image.open('./'+str(i)+'.png').convert('RGB'))
        image = image.astype('int32')
        attackid = teamlabels[i-1]['attackid']
        imageArr[attackid] = image
        imageMax = {}
        imageMin = {}
        for j in range(1,6):
            epsilon = 2**j
            imageMax[j] = np.clip(image + epsilon,0,255).astype('uint8')
            imageMin[j] = np.clip(image - epsilon,0,255).astype('uint8')
        imageMaxArr[attackid] = imageMax
        imageMinArr[attackid] = imageMin

def getdefenseid(attackid):
    for i in range(0,len(teamlabels)):
        if teamlabels[i]["attackid"]==attackid:
            return teamlabels[i]["defenseid"],teamlabels[i]["classlabel"]
    return 0,0

def getclasslabelbydefenseid(defenseid):
    for i in range(0,len(teamlabels)):
        if teamlabels[i]["defenseid"]==defenseid:
            return teamlabels[i]["classlabel"]
    return 0

def getdefensesarray(attackid):
    retArr = []
    for i in range(0,len(teamlabels)):
        if teamlabels[i]["attackid"]==attackid:
            continue
        item = {"defenseid":teamlabels[i]["defenseid"],"classlabel":teamlabels[i]["classlabel"]}
        retArr.append(item)
    return retArr

def isimagevalid(attackid,requestData,epsilon):
    defenseid,classlabel = getdefenseid(attackid)
    if defenseid==0:
        return 0
    
    image = np.array(Image.open(cStringIO.StringIO(requestData)).convert('RGB'))
    image = image.astype('int32')
    epsilonidx = int(math.log(epsilon,2))
    max_clip = imageMaxArr[attackid][epsilonidx]
    min_clip = imageMinArr[attackid][epsilonidx]
    clipped_adv_image = np.clip(image,
                                min_clip,
                                max_clip)
    isSame = (image==clipped_adv_image).all()
    if isSame:
        return 1
    return 0

def savefiletoinput(attackid,targetdefenseid,requestData,myAttackSequence):
    mydefenseid,classlabel = getdefenseid(attackid)
    attackfile = open(inputpath + str(myAttackSequence)+'_'+str(mydefenseid)+'_'+str(targetdefenseid)+'.png', 'wb')
    attackfile.write(requestData)
    attackfile.close()
    resultfilename = str(myAttackSequence)+'_'+str(mydefenseid)+'_'+str(targetdefenseid)+'.txt'
    return resultfilename

def readresultfromoutput(resultfilename):
    try:
        resultfile = open(outputpath+resultfilename,'r')
        lines = resultfile.readlines()
        resultfile.close()
    except Exception,e:
        print 'readresultfromoutput fails ' + str(e)
        return ''
    return lines[0],lines[1]

def waitresult(resultfilename):
    for i in range(0,50):
        if os.path.isfile(outputpath+resultfilename):
            line1,line2 = readresultfromoutput(resultfilename)
            return line1,line2
        else:
            time.sleep(0.1)
    return '',''     
#command handlers start here

#input attackid
#output defense information
def get_mydefense(values):
    try:
        attackid = values["attackid"][0]
        if attackid==None:
            returndata = {"result":2002,"msg":"attackid needed"}
            return json.dumps(returndata)
        else:
            defenseid,classlabel = getdefenseid(attackid)
            if defenseid==0:
                returndata = {"result":2003,"msg":"attackid not found"}
            else:
                returndata = {"result":0,"msg":"getmydefense succeed","defenseid":defenseid,"classlabel":classlabel}            
            return json.dumps(returndata)                       
    except Exception,e:
        print 'Exception get_mydefense: ' + str(e)
        returndata = {"result":2001,"msg":"error while handling getmydefense"}
        return json.dumps(returndata)

def get_defenses(values):
    try:
        attackid = values["attackid"][0]
        if attackid==None:
            returndata = {"result":3002,"msg":"attackid needed"}
            return json.dumps(returndata)
        else:
            defenseid,classlabel = getdefenseid(attackid)
            if defenseid==0:
                returndata = {"result":3003,"msg":"attackid not found"}
            else:
                returndata = getdefensesarray(attackid)
            return json.dumps(returndata)
    except Exception,e:
        print 'Exception get_defenses: ' + str(e)
        returndata =  {"result":3001,"msg":"error while handling getdefenses"}
        return json.dumps(returndata)

def get_teamlogo(values):
    try:
        attackid = values["attackid"][0]
        if attackid==None:
            returndata = {"result":4002,"msg":"attackid needed"}
            return json.dumps(returndata)
        else:
            defenseid,classlabel = getdefenseid(attackid)
            if defenseid==0:
                returndata = {"result":4003,"msg":"attackid not found"}
            else:
                returndata = {"result":0,"msg":"getteamlogo succeed","imagefilename":"./" + str(defenseid) + ".png"}
            return json.dumps(returndata)
    except Exception,e:
        print 'Exception get_teamlogo: ' + str(e)
        returndata =  {"result":4001,"msg":"error while handling getteamlogo"}
        return json.dumps(returndata)

def post_attack(values,requestData):
    global attackSequence,mutex
    
    mutex.acquire()
    attackSequence = attackSequence + 1
    localAttackSequence = attackSequence
    mutex.release()
    try:
        attackid = values["attackid"][0]
        defenseid = int(values["defenseid"][0])
        epsilon = int(values["epsilon"][0])
        if attackid==None or defenseid==None or epsilon==None:
            returndata = {"result":6002,"msg":"attackid, defenseid, epsilon needed"}
            return json.dumps(returndata)
        elif epsilon not in (1,2,4,8,16,32):
            returndata = {"result":6003,"msg":"epsilon need to be 1,2,4,8,16,32"}
            return json.dumps(returndata)
        else:
            mydefenseid,classlabel = getdefenseid(attackid)
            if mydefenseid==0:
                returndata = {"result":6004,"msg":"attackid not found"}
            elif mydefenseid==defenseid:
                returndata = {"result":6005,"msg":"do not attack yourself"}
            else:
                isvalid = isimagevalid(attackid,requestData,epsilon)
                if isvalid==0:
                    returndata = {"result":6006,"msg":"image out of range"}
                else:
                    #add action here
                    print 'image is valid, continue'
                    resultfilename = savefiletoinput(attackid,defenseid,requestData,localAttackSequence)
                    resultclassstr,description = waitresult(resultfilename)
                    print 'got result as: ' + resultclassstr
                    if resultclassstr=='':
                        returndata = {"result":0,"msg":"attack performed","attackresult":"timeout","classlabel":0}
                    else:
                        resultclass = int(resultclassstr)
                        targetlabel = getclasslabelbydefenseid(defenseid)
                        if resultclass==targetlabel:
                            returndata = {"result":0,"msg":"attack performed","attackresult":"positive","classlabel":resultclass,"description":description}
                        else:
                            returndata = {"result":0,"msg":"attack performed","attackresult":"negative","classlabel":resultclass,"description":description}                    
            return json.dumps(returndata)
    except Exception,e:
        print 'Exception post_attack: ' + str(e)
        returndata =  {"result":6001,"msg":"error while handling post_attack"}
        return json.dumps(returndata)
                
#command handlers end here    
    
class TodoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            print 'thread:' + threading.currentThread().getName()
            path = self.path
            writedata = 1
            retstr = ''                        
            query = parse_qs(path[2:])
            print query
            cmd = query['cmd'][0]
            
            if cmd==None:
                retstr = '{"result":1001,"msg":"cmd needed"}'
            elif cmd=='getmydefense':
                retstr = get_mydefense(query)
            elif cmd=='getdefenses':
                retstr = get_defenses(query)
            elif cmd=='getteamlogo':
                retstr = get_teamlogo(query)
                retdata = json.loads(retstr)
                if retdata["result"]==0:
                    writedata = 0
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.end_headers()
                    filename = retdata["imagefilename"]
                    print 'filename=' + filename
                    with open(filename, "rb") as f:
                        bytearr = f.read(1000000)
                        while bytearr != b"":
                            self.wfile.write(bytearr)
                            bytearr = f.read(1000000)
            else:
                retstr = '{"result":1002,"msg":"cmd not supported"}'
                
            if writedata == 1:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(retstr)
        
        except Exception,e:
            print 'Exception do_GET: ' + str(e)
        
    def do_POST(self):
        try:
            print 'thread:' + threading.currentThread().getName()
            path = self.path
            retstr = ''                        
            query = parse_qs(path[2:])
            print query
            cmd = query['cmd'][0]
            length = int(self.headers['content-length'])
            print ('length is: ' + str(length))
            requestData = self.rfile.read(length)            
            
            if cmd==None:
                retstr = '{"result":5001,"msg":"cmd needed"}'
            elif cmd=='postattack':
                retstr = post_attack(query,requestData)
            else:
                retstr = '{"result":5002,"msg":"cmd not supported"}'
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(retstr)
        except Exception,e:
            print 'Exception do_POST: ' + str(e)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
            
if __name__ == '__main__':
    # load original team logo images
    loadteamlogos()
    # Start a simple server, and loop forever
    server = ThreadedHTTPServer(('0.0.0.0', 8866), TodoHandler)
    print("Starting server, use <Ctrl-C> to stop")
    server.serve_forever()
    
