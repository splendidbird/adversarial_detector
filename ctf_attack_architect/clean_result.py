import csv
import pandas as pd
from shutil import *
import datetime
from time import strftime

recordtime = datetime.datetime.now().strftime("%H:%M:%S")
copyfile("target_result.csv", "./history/target_result_deteted_" + recordtime + ".csv")

df = pd.read_csv("target_result.csv")
df.iloc[0:0].to_csv("target_result.csv", index=None)
print("target_result cleaned up, backup saved to history folder")
