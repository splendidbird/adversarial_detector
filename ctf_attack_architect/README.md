# CTF Attack Architect

This is for team NWSec's use during CTF2018

## Commands
Please make sure your current directory is at architect
Please make sure permissons have been given to all .sh files by
```bash
sudo chmod 777 *.sh
```

### Operator Level
[1] load attack

```bash
./load_attack.sh
```
1) acquire attackid, defenseid, attacking IP and port, attacking mission list and orig png
2) generate (renew) orig images for each attacking task in ./dataset/images folder
3) generate (renew) target_class.csv in ./dataset/images folder
4) attackid, defenseid, attacking list, attacking IP and port all renewed and stored in obj "nwsec" of class NWS_Attack

[2] watchdog attack (automatic monitor & attack)

```bash
./watchdog_attack.sh
```
1) start monitoring the folder ./adv_images/
2) when an image appears in this folder:  it will be immediately sent out for attacking based on the info in the filename. 
3) image filename MUST ends with (".png")
4) image filename MUST follow the format "xxxx_epsilon_defenserID.png" in which xxx can be anything allowed in a file name.
5) once attcking executed, 'target_result.csv' file will be automatically updated.
6) the .png file will be deleted from the ./adv_images/ folder.

[3] send all attack (manual full attack)

```bash
./send_attack.sh
```
1) manually send out all ".png" files in the folder ./adv_images/ one by one, based on the info in the filename.
2) after sending out, the file will be deleted from the ./adv_images/ folder.
3) results will be updated in 'target_result.csv' file automatically on image basis (not batch basis)

[4] single-image attack (manual attack with single png)

```bash
./single_attack.sh [required: file-with-full-path]
```

1) manually send out one single ".png" files anywhere (full-path required) for attacking based on the info in the filename.
2) after sending out, the file will be deleted
3) result will be updated in 'target_result.csv' file automatically

[5] display on screen collected results of attacks

```bash
./show_result.sh
```
display on screen the current content of 'target_result.csv'

[6] save current target_result.csv file to history folder and clean up the table

```bash
./clean_result.sh
```
1) save a copy of current 'target_result.csv' file to ./history/ folder (with timeinfo appended to filename)
2) delete all records from current 'target_result.csv' file

[7] print on screen any csv file (with aligned format)

```bash
./printcsv.sh [required: file-with-full-path]
```
display on screen the current content of file-with-full-path, in an aligned way. 
#### Debugging Level
start python2.7 by
$ python
from NWS_Attack import *
