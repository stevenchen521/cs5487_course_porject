
""" CS5487: Course Project - German Credit Analysis """
import logging
import random

import pandas as pd

import numpy as np


logging.basicConfig(level=logging.INFO)

CHK_ACCT ='CHK_ACCT'
DURATION ='DURATION'
HISTORY ='HISTORY'
NEW_CAR='NEW_CAR'
USED_CAR='USED_CAR'
FURNITURE='FURNITURE'
RADIO_TV='RADIO/TV'
EDUCATION='EDUCATION'
RETRAINING='RETRAINING'
BUSINESS='BUSINESS'
DOME_APPLI='DOME_APPLI'
PUR_OTHER='PUR_OTHER'
REPAIR='REPAIR'
AMOUNT='AMOUNT'
SAV_ACCT='SAV_ACCT'
EMPLOYMENT='EMPLOYMENT'
INSTALL_RATE='INSTALL_RATE'
FEMALE_DIV='FEMALE_DIV'
MALE_DIV='MALE_DIV'
MALE_SINGLE='MALE_SINGLE'
MALE_MAR_WID='MALE_MAR_WID'
CO_APPLICANT='CO-APPLICANT'
GUARANTOR='GUARANTOR'
PRESENT_RESIDENT='PRESENT_RESIDENT'
REAL_ESTATE='REAL_ESTATE'
PROP_UNKN_NONE='PROP_UNKN_NONE'
CAR='CAR'
LIFE_INSUR='LIFE_INSUR'
AGE='AGE'
OTHER_INSTALL='OTHER_INSTALL'
RENT='RENT'
OWN_RES='OWN_RES'
FREE='FREE'
NUM_CREDITS='NUM_CREDITS'
JOB='JOB'
NUM_DEPENDENTS='NUM_DEPENDENTS'
TELEPHONE='TELEPHONE'
FOREIGN='FOREIGN'
RESPONSE='RESPONSE'


def load_norm_data(export = './credit_g_normalized.csv'):
    credit_g = pd.read_csv('./credit-g.csv')

    result_columns = [CHK_ACCT, DURATION, HISTORY ,NEW_CAR ,USED_CAR,\
                      FURNITURE,RADIO_TV,EDUCATION,RETRAINING,BUSINESS, \
                      DOME_APPLI,PUR_OTHER, REPAIR, AMOUNT,SAV_ACCT,\
                      EMPLOYMENT, INSTALL_RATE,FEMALE_DIV, MALE_DIV,MALE_SINGLE,\
                      MALE_MAR_WID,CO_APPLICANT,GUARANTOR,PRESENT_RESIDENT,REAL_ESTATE,\
                      PROP_UNKN_NONE, CAR, LIFE_INSUR, AGE,OTHER_INSTALL,	\
                      RENT, OWN_RES,FREE,NUM_CREDITS,JOB, NUM_DEPENDENTS, \
                      TELEPHONE, FOREIGN, RESPONSE]

    result = pd.DataFrame(columns = result_columns)
    
    for index, row in credit_g.iterrows():
        result_row = []

# checking_status
        if(row['checking_status'] == "<0"):
            result_row.append('0')
            # result_row[CHK_ACCT] = 0
        elif(row['checking_status'] == "0<=X<200"):
            result_row.append('1')
        elif(row['checking_status'] == ">=200"):
            result_row.append('2')
        elif(row['checking_status'] == "no checking"):
            result_row.append('3')

        result_row.append(str(row['duration'])) 

# credit_history
        if(row['credit_history'] == "no credits/all paid"):
            result_row.append('0')
        elif(row['credit_history'] == "all paid"):
            result_row.append('1')
        elif(row['credit_history'] == "existing paid"):
            result_row.append('2')
        elif(row['credit_history'] == "delayed previously"):
            result_row.append('3')
        elif(row['credit_history'] == "critical/other existing credit"):
            result_row.append('4')

# Purpose of credit
        if(row['purpose'] == "new car"):
            result_row.append('1')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "used car"):
            result_row.append('0')  #NEW_CAR
            result_row.append('1')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "furniture/equipment"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('1')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "radio/tv"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('1')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "education"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('1')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "retraining"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('1')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "business"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('1')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "domestic appliance"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('1')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "other"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('1')  #PUR_OTHER
            result_row.append('0')  #REPAIR
        elif(row['purpose'] == "repairs"):
            result_row.append('0')  #NEW_CAR
            result_row.append('0')  #USED_CAR
            result_row.append('0')  #FURNITURE
            result_row.append('0')  #RADIO/TV
            result_row.append('0')  #EDUCATION
            result_row.append('0')  #RETRAINING
            result_row.append('0')  #BUSINESS
            result_row.append('0')  #DOME_APPLI
            result_row.append('0')  #PUR_OTHER
            result_row.append('1')  #REPAIR

#credit amount
        result_row.append(str(row['credit_amount']))

# SAV_ACCT
        if(row['savings_status'] == "<100"):
            result_row.append('0')
        elif(row['savings_status'] == "100<=X<500"):
            result_row.append('1')
        elif(row['savings_status'] == "500<=X<1000"):
            result_row.append('2')
        elif(row['savings_status'] == ">=1000"):
            result_row.append('3')
        elif(row['savings_status'] == "no known savings"):
            result_row.append('4')

# EMPLOYMENT
        if(row['employment'] == "unemployed"):
            result_row.append('0')
        elif(row['employment'] == "<1"):
            result_row.append('1')
        elif(row['employment'] == "1<=X<4"):
            result_row.append('2')
        elif(row['employment'] == "4<=X<7"):
            result_row.append('3')
        elif(row['employment'] == ">=7"):
            result_row.append('4')

# INSTALL_RATE
        result_row.append(str(row['installment_commitment']))

# personal_status
        if(row['personal_status'] == "female div/dep/mar"):
            result_row.append('1')  #FEMALE_DIV
            result_row.append('0')  #MALE_DIV
            result_row.append('0')  #MALE_SINGLE
            result_row.append('0')  #MALE_MAR_WID
        elif(row['personal_status'] == "male div/sep"):
            result_row.append('0')  #FEMALE_DIV
            result_row.append('1')  #MALE_DIV
            result_row.append('0')  #MALE_SINGLE
            result_row.append('0')  #MALE_MAR_WID
        elif(row['personal_status'] == "male single"):
            result_row.append('0')  #FEMALE_DIV
            result_row.append('0')  #MALE_DIV
            result_row.append('1')  #MALE_SINGLE
            result_row.append('0')  #MALE_MAR_WID
        elif(row['personal_status'] == "male mar/wid"):
            result_row.append('0')  #FEMALE_DIV
            result_row.append('0')  #MALE_DIV
            result_row.append('0')  #MALE_SINGLE
            result_row.append('1')  #MALE_MAR_WID

# other_parties
        if(row['other_parties'] == "co applicant"):
            result_row.append('1')  #CO-APPLICANT
            result_row.append('0')  #GUARANTOR
        elif(row['other_parties'] == "guarantor"):
            result_row.append('0')  #CO-APPLICANT
            result_row.append('1')  #GUARANTOR
        else:
            result_row.append('0')  #CO-APPLICANT
            result_row.append('0')  #GUARANTOR

# residence_since
        result_row.append(str(row['residence_since']))

# property_magnitude
        if(row['property_magnitude'] == "real estate"):
            result_row.append('1')  #REAL_ESTATE
            result_row.append('0')  #PROP_UNKN_NONE
            result_row.append('0')  #CAR
            result_row.append('0')  #LIFE_INSUR
        elif(row['property_magnitude'] == "no known property"):
            result_row.append('0')  #REAL_ESTATE
            result_row.append('1')  #PROP_UNKN_NONE         
            result_row.append('0')  #CAR
            result_row.append('0')  #LIFE_INSUR
   
        elif(row['property_magnitude'] == "car"):
            result_row.append('0')  #REAL_ESTATE
            result_row.append('0')  #PROP_UNKN_NONE
            result_row.append('1')  #CAR
            result_row.append('0')  #LIFE_INSUR
        elif(row['property_magnitude'] == "life insurance"):
            result_row.append('0')  #REAL_ESTATE
            result_row.append('0')  #PROP_UNKN_NONE
            result_row.append('0')  #CAR
            result_row.append('1')  #LIFE_INSUR

# age
        result_row.append(str(row['age']))


# other_payment_plans

        if(row['other_payment_plans'] == "none"):
            result_row.append('0')
        else:    #(row['other_payment_plans'] == "none"):
            result_row.append('1')

# housing
        if(row['housing'] == "rent"):
            result_row.append('1')  #rent
            result_row.append('0')  #OWN_RES
            result_row.append('0')  #for free
        elif(row['housing'] == "own"):
            result_row.append('0')  #rent
            result_row.append('1')  #OWN_RES
            result_row.append('0')  #for free
        elif(row['housing'] == "for free"):
            result_row.append('0')  #rent
            result_row.append('0')  #OWN_RES
            result_row.append('1')  #for free

#existing_credits
        result_row.append(str(row['existing_credits']))

# JOB
        if(row['job'] == "unemp/unskilled non res"):
            result_row.append('0')
        elif(row['job'] == "unskilled resident"):
            result_row.append('1')
        elif(row['job'] == "skilled"):
            result_row.append('2')
        elif(row['job'] == "high qualif/self emp/mgmt"):
            result_row.append('3')

# num_dependents
        result_row.append(str(row['num_dependents']))

# telephone
        if(row['own_telephone'] == "yes"):
            result_row.append('1')
        elif(row['own_telephone'] == "none"):
            result_row.append('0')

# foreign_worker
        if(row['foreign_worker'] == "yes"):
            result_row.append('1')
        elif(row['foreign_worker'] == "no"):
            result_row.append('0')

# telephone
        if(row['class'] == "good"):
            result_row.append('1')
        elif(row['class'] == "bad"):
            result_row.append('0')




        # result = result.append(result_row)
        if(len(result_row) != 39):
            print(index + result_row)
        result.loc[index] = result_row
    result.to_csv(export)

def main():

    load_norm_data()



if __name__ == '__main__':
    main()

