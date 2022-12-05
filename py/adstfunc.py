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
    
# not yet functioning
# adding participant to file
def adding(name, projectName, age, sex, hearing="", currentSession = 1):

    vorhanden = False
    count = 0

    #failsafe
    if not os.listdir().__contains__(projectName):
        print("given Projectname doesnt exist")
        exit(1)
    try:
        int(age)
    except ValueError:
        print("pls enter an proper age. For example: 23")
        exit(2)

    with open("mastersheet.txt", "r") as master:
        for line in master:
            if line.__contains__(name):
                print ("found: "+line)
                count =+ 1

    if count == 0:
        #TODO
        # point of breaking -> while writing this function seperated from adstfunc.py
        # if later on added to adsfunc change adstfunc.randId() to just randId()
        

        id = adstfunc.randId()
        with open("mastersheet.txt", "w") as file:
            file.write(name + " "+id)

        with open("particpants.csv", "w", newline='') as csv:
            writer = csv.writer(csv,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL
                                )

            writer.writerow([id, age, sex, hearing])

    if count == 1:
        with open("mastersheet.txt", "r") as r:
            for line in r:
                if name == line.split(' ')[1]:
                    id = line.split(' ')[2]

if __name__ == "__main__":
    init("MSL")
