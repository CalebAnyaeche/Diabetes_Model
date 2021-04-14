# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Early Stage Diabetes Risk Prediction Dataset

# Team Members: Caleb Anyaeche and Matthew Zlibut

# #### Background 
#
# Diabetes is one of the fastest-growing chronic life-threatening diseases that have already affected 422 million people worldwide according to the report of the World Health Organization (WHO), in 2018. Because of the presence of a generally long asymptomatic stage, early detection of diabetes is constantly wanted for a clinically significant result. Around 50% of all people suffering from diabetes are undiagnosed because of its long-term asymptomatic phase. The early diagnosis of diabetes is only possible by proper assessment of both common and less common sign symptoms, which could be found in different phases from disease initiation up to diagnosis. Data mining classification techniques have been well accepted by researchers for the risk prediction model of the disease.

# #### Goal
#
# The objective is to build a machine learning based model to predict if a patient has or will have a early stage diabetes risk.

# #### Reference Paper Result
#
# Since one has to pay to see the paper referenced, the eventual results are not known. However, it referenced that Random Forest Algorithm was found to have the best accuracy.

# ## Dataset Information:

# #### Dependent attribute
#
# Class (1.Positive, 2.Negative)

# #### Dropped columns 
#
# No column needs to be dropped, as neither of them are keys nor can be derived from others.

# #### Numerical attribute
#
# Age (20-65)

# #### ordinal attributes
#
# No column is ordinal categorical.

# #### Nominal attributes 
#
# All other columns but Age are Nominal categorical.

# #### Missing Data
#
# The dataset has no missing data.

import pandas as pd
import numpy as np

print(df.shape)
df.head()

# +
df = pd.read_csv('diabetes_data_upload.csv')

df.isna().sum()
# -

df.info()
