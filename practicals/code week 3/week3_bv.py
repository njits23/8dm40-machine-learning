from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pandas as pd


gene_expression = pd.read_csv("./data/RNA_expression_curated.csv", sep=',', header=0, index_col=0)
drug_response = pd.read_csv("./data/drug_response_curated.csv", sep=',', header=0, index_col=0)
