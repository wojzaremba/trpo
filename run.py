import os

files = ["main_copy", "main_duplicated", "main_reverse"]

os.system("rm logs_*")
os.system("k screen")
os.system("screen -wipe")
gpuidx = 0


for f in files:
    os.system("screen -dm -S %s bash -c '. ~/.profile; python %s.py 2>&1 | tee logs_%s ; bash'" % (f, f, f))
