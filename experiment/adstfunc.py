import string
import random
import os
import csv
import slab

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
    """

    :param projectName: only parameter is the project name
    :return: creates foldersystem with resultsdirectory and soundsamples, Furthermore documentation is set up
    """
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
def rFile(name, projectName, age='', sex='', hearing=''):
    """

    :param name: Name of the subject
    :param projectName: name of the project
    :param age: age of the subject
    :param sex: sex of the subject
    :param hearing: anything hearing related that needs to be known about subject
    :return: tries to find valid subject "targets" with given parameters and prepare slab.resultFile() method,
    if none exist it will try to create given participant with an random Id ,
    if unable to single out one participant manuell input is required.
    """
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

    os.chdir(os.getcwd()+"/results")
    with open("mastersheet.txt", "r") as master:    # checking if the participant is in the mastersheet
        for line in master:
            if line.__contains__(name):             # checks if given name is in file
                print("found: "+line)               # prints valid targets
                count += 1                          # counts valid targets for desicion based on amount of targets

    if count == 0:
        #TODO
        # point of breaking -> while writing this function seperated from adstfunc.py
        # if later on added to adsfunc change adstfunc.randId() to just randId()

        id = randId()  #creates new id
        with open("mastersheet.txt", "w") as file:  # fills database with new participant
            file.write(name + " "+id)

        # TODO
        # possible point of break "writer" not found

        with open("particpants.csv", "w", newline='') as csv:
            writer = csv.writer(csv,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL
                                )

            writer.writerow([id,
                             age,
                             sex,
                             hearing]
                            )
        os.chdir(os.getcwd())
        print("returned Resultsfile you are now able to add or read")
        return slab.ResultsFile(id, os.getcwd()+"/results")

    if count == 1:              # returns the one valid participant as resultFile
        with open("mastersheet.txt", "r") as r:
            for line in r:
                if name == line.split(' ')[1]:
                    id = line.split(' ')[2]
        print("returned Resultsfile you are now able to add or read")
        return slab.ResultsFile(id, os.getcwd()+"/results")

    if count > 1:  # trying to reduce amount of valid targets with bigger search
        with open("mastersheet.txt", "r") as master:
            for line in master:
                if line.__contains__(name):
                    id = line.split(" ")[2]
                    with open ("participants.csv", mode='r') as csv:
                        if line.__contains__(id):
                            if age != line.split(" ")[2]:
                                count -= 1

        if count == 1:  # returns the one valid participant as resultFile
            with open("mastersheet.txt", "r") as r:
                for line in r:
                    if name == line.split(' ')[1]:
                        id = line.split(' ')[2]
            print("returned Resultsfile you are now able to add or read")
            return slab.ResultsFile(id, os.getcwd() + "/results")

        else:  # if unable to reduce count, manuell input is required
            print("unable to reduce count")
            id = input("pls enter the wanted id\n check the csv file for the right sex and age!")
            return slab.ResultsFile(id, os.getcwd()+"/results")

if __name__ == "__main__":
    init("MSL")
