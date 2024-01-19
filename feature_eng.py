import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#load data from csv into dataframe
data = pd.read_csv('data.csv', delimiter=';')

X = data.drop('G3', axis =1)
y = data['G3']

#create separate set for no grades regression
X_no_grades = X.drop(['G1', 'G2'], axis = 1)

#manually binarize bc the column transformer doesn't like label binarizer 
lb = LabelBinarizer()
for x in  ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid','activities','nursery','internet','higher','romantic']:
    X[x] = lb.fit_transform(X[x])
    X_no_grades[x] = lb.fit_transform(X_no_grades[x])

#engineered features
X["Pedu"] = (X['Medu'] + X['Fedu'])/2 #parent education avg
X['Talc'] = (X['Dalc'] + X['Walc'])/2 #total alc consumption 
X["Gavg"] = (X['G1'] + X['G2'])/2 #avg g1 and g2 grade
X['social'] = X['goout'] + X['romantic'] + X['famrel'] #sum of goout and romantic and famrel, max value of 11

#engineered features for no_grade
X_no_grades["Pedu"] = X["Pedu"]
X_no_grades['Talc']=X['Talc'] 
X_no_grades['social'] = X['social']

#column transformers
ohe = OneHotEncoder(drop ='first')
oe = OrdinalEncoder()
ohe_trans = ColumnTransformer([('onehot', ohe, ['Mjob', 'Fjob', 'reason', 'guardian'])],
                                    remainder = 'passthrough')
ord_trans = ColumnTransformer([('ord', oe, ['Mjob', 'Fjob', 'reason', 'guardian'])],
                                    remainder = 'passthrough')

#split and transform training and testing set
def split_transform(x, y, transformer):
    #split dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state= 1)
    X_train = pd.DataFrame(transformer.fit_transform(X_train), columns = [s.split('__')[1] for s in transformer.get_feature_names_out()])
    X_test = pd.DataFrame(transformer.fit_transform(X_test), columns = [s.split('__')[1] for s in transformer.get_feature_names_out()])
    return X_train, X_test, y_train, y_test


