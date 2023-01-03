import streamlit as st
import pandas as pd
from default.data_distribution import *
from default.machine_learning_models import *
import matplotlib.pyplot as plt
from default.read_defaults import *
from sklearn.preprocessing import PolynomialFeatures

with st.sidebar:
    splitPercent = st.slider('Chose percentage of data for test', 0, 100, 20)/100

    st.write('Type and how much augumentation - to be added')

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['General', 'Decision tree', 'Naive Bayes', 'Random forest', 'Logistic regression', 'SVM', 'Machine learning on PCA'])
selectedColumns = {}

for c in data_basic.columns:
   selectedColumns[c] = True

with st.sidebar:
   with st.expander("Select columns"):
      for c in data_basic.columns:
          if(c != 'target'):
            selectedColumns[c] = st.checkbox(c, selectedColumns[c])
   filteredColumns = [k[0] for k in selectedColumns.items() if (k[1] is True and k[0] != 'target')]

   data_filtered = data_basic[filteredColumns]

   with st.expander('Augmentation'):
        degree = st.slider('How many new samples', 0, 5, 2)
        poly = PolynomialFeatures(degree=degree)
        data_filtered = pd.DataFrame(poly.fit_transform(data_filtered))
        data_filtered['target'] = data_basic['target']
        n = st.slider('How many new samples', 0, 300, 20)
        sampled_data = data_basic.sample(n=n, replace=False)
        data_filtered.append(sampled_data, ignore_index=True)
        y_basic = data_filtered['target']
        X_basic = data_filtered

X_train, X_test, y_train, y_test = splitData(X_basic, y_basic, splitPercent)

with tab1:
    st.write(texts["machineLearningAbstract"])

with tab2:
    col21, col22 = st.columns(2)
    with col21:
        st.write(texts["decisionTreeAbstract"])
    with col22:
        DT_basic_matrix, DT_basic_acc = getPrediction('Decision tree', X_train, X_test, y_train, y_test)

        st.dataframe(DT_basic_matrix)
        st.write(DT_basic_acc)

        labels = 'True negatives', 'False negative', 'False positive', 'True positive'
        sizes = [DT_basic_matrix[0][0], DT_basic_matrix[0][1], DT_basic_matrix[1][0], DT_basic_matrix[1][1]]
        explode = (0.1, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,  labels=labels, explode=explode, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)

with tab3:
    col21, col22 = st.columns(2)
    with col21:
        st.write(texts["nativeBayesAbstract"])
    with col22:
        DT_basic_matrix, DT_basic_acc = getPrediction('Naive Bayes', X_train, X_test, y_train, y_test)

        st.dataframe(DT_basic_matrix)
        st.write(DT_basic_acc)

        labels = 'True negatives', 'False negative', 'False positive', 'True positive'
        sizes = [DT_basic_matrix[0][0], DT_basic_matrix[0][1], DT_basic_matrix[1][0], DT_basic_matrix[1][1]]
        explode = (0.1, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,  labels=labels, explode=explode, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)

with tab4:
    col21, col22 = st.columns(2)
    with col21:
        st.write(texts["randomForestAbstract"])
    with col22:
        DT_basic_matrix, DT_basic_acc = getPrediction('Random forest', X_train, X_test, y_train, y_test)

        st.dataframe(DT_basic_matrix)
        st.write(DT_basic_acc)

        labels = 'True negatives', 'False negative', 'False positive', 'True positive'
        sizes = [DT_basic_matrix[0][0], DT_basic_matrix[0][1], DT_basic_matrix[1][0], DT_basic_matrix[1][1]]
        explode = (0.1, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,  labels=labels, explode=explode, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)

with tab5:
    col21, col22 = st.columns(2)
    with col21:
        st.write(texts["logisticRegressionAbstract"])
    with col22:
        DT_basic_matrix, DT_basic_acc = getPrediction('Logistic regression', X_train, X_test, y_train, y_test)

        st.dataframe(DT_basic_matrix)
        st.write(DT_basic_acc)

        labels = 'True negatives', 'False negative', 'False positive', 'True positive'
        sizes = [DT_basic_matrix[0][0], DT_basic_matrix[0][1], DT_basic_matrix[1][0], DT_basic_matrix[1][1]]
        explode = (0.1, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,  labels=labels, explode=explode, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)

with tab6:
    col21, col22 = st.columns(2)
    with col21:
        st.write(texts["SVMAbstract"])
    with col22:
        DT_basic_matrix, DT_basic_acc = getPrediction('SVM', X_train, X_test, y_train, y_test)

        st.dataframe(DT_basic_matrix)
        st.write(DT_basic_acc)

        labels = 'True negatives', 'False negative', 'False positive', 'True positive'
        sizes = [DT_basic_matrix[0][0], DT_basic_matrix[0][1], DT_basic_matrix[1][0], DT_basic_matrix[1][1]]
        explode = (0.1, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,  labels=labels, explode=explode, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)

with tab7:
    st.write('Take two main components so data can be visualised on scatterplot and where it would be divided.')
