import numpy as np
import csv
import random
import math


GRE=[]
TOEFL=[]
URating=[]
CGPA=[]
Chance=[]

GRE_training=[]
TOEFL_training=[]
URating_training=[]
CGPA_training=[]
Chance_training=[]

GRE_cross=[]
TOEFL_cross=[]
URating_cross=[]
CGPA_cross=[]
Chance_cross=[]

GRE_test=[]
TOEFL_test=[]
URating_test=[]
CGPA_test=[]
Chance_test=[]

def calculoDeErrorCross(modelo):
    pronosticoList=[]
    
    for x in np.matmul(XCross, modelo):
        pronosticoList.append(x[0].item())
        
    
    sumatoria=0
    for j in range(len(Chance_cross)):
        sumatoria+=(abs(pronosticoList[j]-float(Chance_cross[j]))*100)/float(Chance_cross[j])
    
    return (sumatoria/(len(Chance_cross)))

    
def calculoDeErrorTest(modelo):
    pronosticoList=[]
    
    for x in np.matmul(XTest, modelo):
        pronosticoList.append(x[0].item())
        
    
    sumatoria=0
    for j in range(len(Chance_test)):
        sumatoria+=(abs(pronosticoList[j]-float(Chance_test[j]))*100)/float(Chance_cross[j])
    
    return (sumatoria/(len(Chance_test)))


with open('Admission_Predict.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            GRE.append(row[1])
            TOEFL.append(row[2])
            URating.append(row[3])
            CGPA.append(row[6])
            Chance.append(row[8])
            line_count += 1


random.seed(30)
contador=0
for i in random.sample(range(400),400):
    if(contador<240):
        #Training
        GRE_training.append(float(GRE[i]))
        TOEFL_training.append(float(TOEFL[i]))
        URating_training.append(float(URating[i]))
        CGPA_training.append(float(CGPA[i]))
        Chance_training.append(float(Chance[i]))
    elif(contador<320):
        #Cross
        GRE_cross.append(float(GRE[i]))
        TOEFL_cross.append(float(TOEFL[i]))
        URating_cross.append(float(URating[i]))
        CGPA_cross.append(float(CGPA[i]))
        Chance_cross.append(float(Chance[i]))
    else:
        #Test
        GRE_test.append(float(GRE[i]))
        TOEFL_test.append(float(TOEFL[i]))
        URating_test.append(float(URating[i]))
        CGPA_test.append(float(CGPA[i]))
        Chance_test.append(float(Chance[i]))

    contador+=1
    


#for i in range(len(GRE_training)):
 #   print(str(GRE_training[i])+"\t-\t"+str(TOEFL_training[i])+"\t-\t"+str(URating_training[i])+"\t-\t"+str(CGPA_training[i])+"\t-\t"+str(Chance_training[i]))


TRAINING_ELEMENTS = 240

X = np.vstack(
    (
        np.ones(len(GRE_training)),
        (np.array(GRE_training)**2)/200,
        (np.array(TOEFL_training)**3)/1500,
    )
).T

XBad = np.vstack(
    (
        np.ones(len(GRE_training)),
        (np.array(GRE_training)),
        (np.array(TOEFL_training)),
    )
).T

XCross = np.vstack(
    (
        np.ones(len(GRE_cross)),
        (np.array(GRE_cross)**2)/200,
        (np.array(TOEFL_cross)**3)/1500,
    )
).T

XTest = np.vstack(
    (
        np.ones(len(GRE_test)),
        (np.array(GRE_test)**2)/200,
        (np.array(TOEFL_test)**3)/1500,
    )
).T

y =np.array(Chance_training)
dataset_1 = (X, y.reshape(TRAINING_ELEMENTS, 1))

