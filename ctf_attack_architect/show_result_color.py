import csv
from termcolor import colored, cprint

with open("target_result.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    print('\t'.join(header))
    while(True):
        try:
            row = next(reader)
        except:
            break
        if row[5]=='pos':
            print colored('\t'.join(row), 'green')
        elif row[2]!=row[4]:
            print colored('\t'.join(row), 'yellow')
        else:
            print colored('\t'.join(row), 'red')
