import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import researchpy as rp
from sympy.stats.sampling.sample_scipy import scipy
import statistics
import altair as alt

from default.read_defaults import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

@st.cache
def getPca(X_basic):
   pca = PCA()
   transformed_data = pd.DataFrame(pca.fit_transform(X_basic.values))
   explained_variance_ratio = pca.explained_variance_ratio_
   return explained_variance_ratio, transformed_data

@st.cache
def transformData(degree):
   poly = PolynomialFeatures(degree=degree)
   return poly.fit_transform(X_basic.values)

y_basic = data_basic['target']
X_basic = data_basic.copy()
X_basic = X_basic.drop(['target'], axis=1)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["T-test", "Boxplot", "Correlation", "PCA", "Augumentation"])
selectedColumns = {}

for c in X_basic.columns:
   selectedColumns[c] = True

with st.sidebar:
   with st.expander("Select columns"):
      for c in X_basic.columns:
         selectedColumns[c] = st.checkbox(c, selectedColumns[c])

filteredColumns = [k[0] for k in selectedColumns.items() if (k[1] is True)]

with tab1:
   col21, col22 = st.columns(2)
   with col21:
      st.write(texts['t-testAbstract'])
   with col22:
      selectSec = filteredColumns.copy()
      select1 = st.selectbox('Select column: ', set(filteredColumns), key='t-test-slider')
      zeroSide = data_basic[select1][data_basic['target']==0]
      oneSide = data_basic[select1][data_basic['target']==1]

      summary, results = scipy.stats.ttest_ind(zeroSide, oneSide)
      st.write(summary)

with tab2:
   col21, col22 = st.columns(2)
   with col21:
      st.write(texts['boxplotAbstract'])
   with col22:
      i = 0
      selectSec = filteredColumns.copy()
      select1 = st.selectbox('Select column: ', set(filteredColumns), key='box-plot-abstract')
      fig, ax = plt.subplots()
      ax.boxplot(data_basic[select1])
      ax.set_title(select1)
      st.pyplot(fig)

with tab3:
   st.write(texts['scatterCorrAbstract'])
   numOfScatter = int(len(filteredColumns)*(len(filteredColumns)-1)/2)
   col1, col2 = st.columns(2)
   selectSec = filteredColumns.copy()
   with col1:
      select1 = st.selectbox('Select column: ', set(selectSec), key='select-scatter-1')
      filteredColumns.remove(select1)
   with col2:
      select2 = st.selectbox('Select column: ', set(filteredColumns), key='select-scatter-2')
   fig, ax = plt.subplots()
   corr = np.corrcoef(data_basic[select1], data_basic[select2])
   ax.scatter(data_basic[select1], data_basic[select2])
   ax.set_title(corr[0, 1])
   ax.set_xlabel(select1)
   ax.set_ylabel(select2)
   st.pyplot(fig)

with tab4:
   st.write(texts['PCAAbstract'])

   explained_variance_ratio, transformed_data = getPca(X_basic)
   col1, col2 = st.columns(2)
   with col1:
      for i in range(len(explained_variance_ratio)):
         st.markdown("{num}. component explains <font color='green'>{perc}%</font> of data variation.".format(num=i + 1, perc=str(round( explained_variance_ratio[i] * 100, 2))), unsafe_allow_html=True)

   with col2:
      numOfcomponents = st.slider('Number of components', 2, 4)
      pca_x = 'component 1'
      pca_y = 'component 2'
      pca_size = 'component 3'
      pca_color = 'component 4'
      chart_data = transformed_data.iloc[:, :numOfcomponents]
      if(numOfcomponents == 2):
         chart_data.columns = [pca_x, pca_y]
         c = alt.Chart(chart_data).mark_circle().encode(
            x=pca_x, y=pca_y, tooltip=[pca_x, pca_y])
         st.altair_chart(c, use_container_width=True)
      if (numOfcomponents == 3):
         chart_data.columns = [pca_x, pca_y, pca_size]
         c = alt.Chart(chart_data).mark_circle().encode(
            x=pca_x, y=pca_y, size=pca_size, tooltip=[pca_x, pca_y, pca_size])
         st.altair_chart(c, use_container_width=True)
      if (numOfcomponents == 4):
         chart_data.columns = [pca_x, pca_y, pca_size, pca_color]
         c = alt.Chart(chart_data).mark_circle().encode(
            x=pca_x, y=pca_y, size=pca_size, color=pca_color, tooltip=[pca_x, pca_y, pca_size, pca_color])
         st.altair_chart(c, use_container_width=True)
      #st.write(transformed_data.iloc[:, :numOfcomponents])

with tab5:
   st.write(texts['augmentationAbstract'])
   with st.expander('Polynominal features'):
     st.write(texts['polyAbstract'])
     degree = st.slider('how many degrees should be added?', 0, 5, 2)
     augmented_data = transformData(degree)
     st.write(augmented_data)
   with st.expander('New samples'):
     n = st.slider('How many new samples?', 0, 300, 20)
     sampled_data = data_basic.sample(n=n, replace=False)
     st.write(sampled_data)
     st.write(len(sampled_data))
