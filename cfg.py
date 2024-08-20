import string
import os
import nltk
import json
import sys
nltk.download ('stopwords')
nltk.download ('punkt')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from joblib import load
import joblib

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0    