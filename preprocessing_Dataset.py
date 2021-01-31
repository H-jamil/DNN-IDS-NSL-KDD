import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
#Libraries for printing tables in readable format
from tabulate import tabulate

#Library for creating an excel sheet
import xlsxwriter
import operator
import seaborn as sns
import joblib

#Feature selection library
from featureselectionlibrary import featureSelectionUsingTheilU
from featureselectionlibrary import featureSelectionUsingChisquaredTest
from featureselectionlibrary import featureSelectionUsingRandomForestClassifier
from featureselectionlibrary import featureSelectionUsingExtraTreesClassifier

#feature encoding library
from featureencodinglibrary import featureEncodingUsingOneHotEncoder
from featureencodinglibrary import featureEncodingUsingLabelEncoder
from featureencodinglibrary import featureEncodingUsingBinaryEncoder
from featureencodinglibrary import featureEncodingUsingFrequencyEncoder

#feature scaling library
from featurescalinglibrary import featureScalingUsingMinMaxScaler
from featurescalinglibrary import featureScalingUsingStandardScalar
from featurescalinglibrary import featureScalingUsingBinarizer
from featurescalinglibrary import featureScalingUsingNormalizer
#feature scaling Library
from classificationlibrary import classifyUsingDecisionTreeClassifier
from classificationlibrary import classifyUsingLogisticRegression
from classificationlibrary import classifyUsingLinearDiscriminantAnalysis
from classificationlibrary import classifyUsingGaussianNB
from classificationlibrary import classifyUsingRandomForestClassifier
from classificationlibrary import classifyUsingExtraTreesClassifier
from classificationlibrary import classifyUsingKNNClassifier
from classificationlibrary import findingOptimumNumberOfNeighboursForKNN



def getPathToTrainingAndTestingDataSets():
	trainingFileNameWithAbsolutePath = "/Users/jamil/Desktop/IDS-NSL-KDD/Autoencoder/KDDTrain+_20Percent.csv"
	testingFileNameWithAbsolutePath = "/Users/jamil/Desktop/IDS-NSL-KDD/Autoencoder/KDDTest-21.csv"
	return trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath

def loadCSV (fileNameWithAbsolutePath):
    dataSet = pd.read_csv(fileNameWithAbsolutePath)
    return dataSet

def getLabelName():
	return 'attack_type'

def printList (list,heading):
    for i in range(0, len(list)):
        list[i] = str(list[i])
    if len(list)>0:
        print(tabulate([i.strip("[]").split(", ") for i in list], headers=[heading], tablefmt='orgtbl')+"\n")


#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    anyMissingValuesInTheDataset = dataSet.isnull().values.any()
    return anyMissingValuesInTheDataset

#This function is used to check for duplicate records in a given dataSet
#if duplicate are present it will
def checkForDulicateRecords (dataSet):
    totalRecordsInDataset = len(dataSet.index)
    numberOfUniqueRecordsInDataset = len(dataSet.drop_duplicates().index)
    anyDuplicateRecordsInTheDataset = False if totalRecordsInDataset == numberOfUniqueRecordsInDataset else True
    print('Total number of records in the dataset: {}\nUnique records in the dataset: {}'.format(totalRecordsInDataset,numberOfUniqueRecordsInDataset))
    return anyDuplicateRecordsInTheDataset

#Split the complete dataSet into training dataSet and testing dataSet
def splitCompleteDataSetIntoTrainingSetAndTestingSet(completeDataSet):
	labelName = getLabelName()
	label = completeDataSet[labelName]
	features = completeDataSet.drop(labelName,axis=1)
	featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet=train_test_split(features,label,test_size=0.4, random_state=42)
	print("features.shape: ",features.shape)
	print("label.shape: ",label.shape)
	return featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet




def getStatisticsOfData (dataSet):
    print("***** Start checking the statistics of the dataSet *****\n")

    labelName = getLabelName()
    #Number of rows and columns in the dataset
    print("***** Shape (number of rows and columns) in the dataset: ", dataSet.shape)

    #Total number of features in the dataset
    numberOfColumnsInTheDataset = len(dataSet.drop([labelName],axis=1).columns)
    #numberOfColumnsInTheDataset = len(dataSet.columns)
    print("***** Total number of features in the dataset: ",numberOfColumnsInTheDataset)

    #Total number of categorical featuers in the dataset
    categoricalFeaturesInTheDataset = list(set(dataSet.drop([labelName],axis=1).columns) - set(dataSet.drop([labelName],axis=1)._get_numeric_data().columns))
    #categoricalFeaturesInTheDataset = list(set(dataSet.columns) - set(dataSet._get_numeric_data().columns))
    print("***** Number of categorical features in the dataset: ",len(categoricalFeaturesInTheDataset))

    #Total number of numerical features in the dataset
    numericalFeaturesInTheDataset = list(dataSet.drop([labelName],axis=1)._get_numeric_data().columns)
    #numericalFeaturesInTheDataset = list(dataSet._get_numeric_data().columns)
    print("***** Number of numerical features in the dataset: ",len(numericalFeaturesInTheDataset))

    #Names of categorical features in the dataset
    print("\n***** Names of categorical features in dataset *****\n")
    printList(categoricalFeaturesInTheDataset,'Categorical features in dataset')

    #Names of numerical features in the dataset
    print("\n***** Names of numerical features in dataset *****\n")
    printList(numericalFeaturesInTheDataset,'Numerical features in the dataset')

    #Checking for any missing values in the data set
    anyMissingValuesInTheDataset = checkForMissingValues(dataSet)
    print("\n***** Are there any missing values in the data set: ", anyMissingValuesInTheDataset)

    anyDuplicateRecordsInTheDataset = checkForDulicateRecords(dataSet)
    print("\n***** Are there any duplicate records in the data set: ", anyDuplicateRecordsInTheDataset)
    #Check if there are any duplicate records in the data set
    if (anyDuplicateRecordsInTheDataset):
        dataSet = dataSet.drop_duplicates()
        print("Number of records in the dataSet after removing the duplicates: ", len(dataSet.index))

    #How many number of different values for label that are present in the dataset
    print('\n****** Number of different values for label that are present in the dataset: ',dataSet[labelName].nunique())
    #What are the different values for label in the dataset
    print('\n****** Here is the list of unique label types present in the dataset ***** \n')
    printList(list(dataSet[getLabelName()].unique()),'Unique label types in the dataset')

    #What are the different values in each of the categorical features in the dataset
    print('\n****** Here is the list of unique values present in each categorical feature in the dataset *****\n')
    categoricalFeaturesInTheDataset = list(set(dataSet.columns) - set(dataSet._get_numeric_data().columns))
    numericalFeaturesInTheDataset = list(dataSet._get_numeric_data().columns)
    for feature in categoricalFeaturesInTheDataset:
        uniq = np.unique(dataSet[feature])
        print('\n{}: {} '.format(feature,len(uniq)))
        printList(dataSet[feature].unique(),'distinct values')

    print('\n****** Label distribution in the dataset *****\n')
    print(dataSet[labelName].value_counts())
    print()

    print("\n***** End checking the statistics of the dataSet *****")


def defineArrayForPreProcessing():
	arrayOfModels = [
		[
			"ExtraTreesClassifier",
			"OneHotEncoder",
			"Standardization",
		]
	]
	print(arrayOfModels)
	return arrayOfModels

def performPreprocessing(trainingDataSet, testingDataSet, arrayOfModels):
    for i in range(0,len(arrayOfModels)):
        print('***************************************************************************************************************************')
        print('********************************************* Building Model-', i ,' As Below *************************************************')
        print('\t -- Feature Selection: \t ', arrayOfModels[i][0], ' \n\t -- Feature Encoding: \t ', arrayOfModels[i][1], ' \n\t -- Feature Scaling: \t ', arrayOfModels[i][2], '\n')

        trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath = getPathToTrainingAndTestingDataSets()
        trainingDataSet = loadCSV(trainingFileNameWithAbsolutePath)
        testingDataSet = loadCSV(testingFileNameWithAbsolutePath)

        labelName = getLabelName()
        label = trainingDataSet[labelName]

        #Combining the test and training datasets for preprocessing together, because we observed that in some datasets
        #the values in the categorical columns in test dataset and train dataset are being different this causes issues while
        #applying classification techniques
        completeDataSet = pd.concat(( trainingDataSet, testingDataSet ))

        #difficultyLevel = completeDataSet.pop('difficulty_level')

        print("completeDataSet.shape: ",completeDataSet.shape)
        print("completeDataSet.head: ",completeDataSet.head())

        #Feature Selection
        if arrayOfModels[i][0] == 'TheilsU':
            #Perform feature selection using TheilU
            completeDataSetAfterFeatuerSelection = featureSelectionUsingTheilU(completeDataSet)
        elif arrayOfModels[i][0] == 'Chi-SquaredTest':
            #Perform feature selection using Chi-squared Test
            completeDataSetAfterFeatuerSelection = featureSelectionUsingChisquaredTest(completeDataSet)
        elif arrayOfModels[i][0] == 'RandomForestClassifier':
            #Perform feature selection using RandomForestClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingRandomForestClassifier(completeDataSet)
        elif arrayOfModels[i][0] == 'ExtraTreesClassifier':
            #Perform feature selection using ExtraTreesClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingExtraTreesClassifier(completeDataSet)

        #Feature Encoding
        if arrayOfModels[i][1] == 'LabelEncoder':
            #Perform lable encoding to convert categorical values into label encoded features
            completeEncodedDataSet = featureEncodingUsingLabelEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'OneHotEncoder':
            #Perform OnHot encoding to convert categorical values into one-hot encoded features
            completeEncodedDataSet = featureEncodingUsingOneHotEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'FrequencyEncoder':
            #Perform Frequency encoding to convert categorical values into frequency encoded features
            completeEncodedDataSet = featureEncodingUsingFrequencyEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'BinaryEncoder':
            #Perform Binary encoding to convert categorical values into binary encoded features
            completeEncodedDataSet = featureEncodingUsingBinaryEncoder(completeDataSetAfterFeatuerSelection)

        #Feature Scaling
        if arrayOfModels[i][2] == 'Min-Max':
            #Perform MinMaxScaler to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingMinMaxScaler(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Binarizing':
            #Perform Binarizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingBinarizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Normalizing':
            #Perform Normalizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingNormalizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Standardization':
            #Perform Standardization to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingStandardScalar(completeEncodedDataSet)

        #Split the complete dataSet into training dataSet and testing dataSet
        featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet = splitCompleteDataSetIntoTrainingSetAndTestingSet(completeEncodedAndScaledDataset)

        trainingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTrainingDataSet, labelInPreProcessedTrainingDataSet], axis=1, sort=False)
        testingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTestingDataSet, labelInPreProcessedTestingDataSet], axis=1, sort=False)

    return 	completeEncodedAndScaledDataset

trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath = getPathToTrainingAndTestingDataSets()

trainingDataSet = loadCSV(trainingFileNameWithAbsolutePath)
difficultyLevel = trainingDataSet.pop('difficulty_level')
labelName = getLabelName()
label = trainingDataSet[labelName]
getStatisticsOfData(trainingDataSet)
print("\n***** Here is how training dataset looks like before performing any pre-processing *****")
print("\n#########\n")
print(trainingDataSet.head())

#Define file names and call loadCSV to load the CSV files
testingDataSet = loadCSV(testingFileNameWithAbsolutePath)
difficultyLevel = testingDataSet.pop('difficulty_level')
getStatisticsOfData(testingDataSet)
print("\n***** Here is how to testing dataset looks like before performing any pre-processing *****")
testingDataSet.head()

arrayOfModels = defineArrayForPreProcessing()
completeEncodedAndScaledDataset = performPreprocessing(trainingDataSet, testingDataSet, arrayOfModels)
print(completeEncodedAndScaledDataset.head())
