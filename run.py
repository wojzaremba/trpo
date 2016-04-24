import os

tasks = ["Copy-v0", "DuplicatedInput-v0", "Reverse-v0"]

os.system("rm logs_*")
os.system("k screen")
os.system("screen -wipe")


for t in tasks:
    os.system("screen -dm -S %s bash -c '. ~/.profile; python main.py %s 2>&1 | tee logs_%s ; bash'" % (t, t, t))
