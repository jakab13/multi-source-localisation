import string
import random
import os
import csv

def randId():
    first = input("enter if the participant is visiting for the first time.\n")
    if first.__contains__("y"):
        print("understood yes")
        # generating new participant id
        id = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(6)])

        return id

    elif first.__contains__("n"):
        print("understood no\npls enter the id manually")
        id = input("What is the id?\n")
        return id
        #asking for manuell imput, maybe setting up a file system for automatic load/read
    else:
        print("input is invalid, pls answer with either yes or no")
        exit(1)

def init(projectName):
    os.mkdir(os.getcwd()+"/"+projectName)  # creating Projectfolder
    os.chdir(os.getcwd()+"/"+projectName)  # changing Location to Project

    os.mkdir(os.getcwd()+"/results")       # adding resultsfolder
    os.mkdir(os.getcwd()+"/soundSamples")  # adding soundfolder

    os.chdir(os.getcwd() + "/results")     # changing location to resultsfolder
    with open('participants.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL
                            )
        writer.writerow(['participant_ID', 'age', 'sex', 'Hearing_related'])
    print("set up finished")
