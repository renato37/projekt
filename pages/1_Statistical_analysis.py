import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import researchpy as rp
from default.read_defaults import *
import numpy as np
from sklearn.decomposition import PCA

y_basic = data_basic['target']

tab1, tab2, tab3, tab4, tab5 = st.tabs(["T-test", "Boxplot", "Scatterplot", "PCA", "Augumentation"])
selectedColumns = {}

for c in data_basic.columns:
   selectedColumns[c] = True

with st.sidebar:
   with st.expander("Select columns"):
      for c in data_basic.columns:
         selectedColumns[c] = st.checkbox(c, selectedColumns[c])

filteredColumns = [k[0] for k in selectedColumns.items() if (k[1] is True)]

with tab1:
   col21, col22 = st.columns(2)
   with col21:
      st.write(texts['t-testAbstract'])
   with col22:
      selectSec = filteredColumns.copy()
      selectSec.remove('target')
      select1 = st.selectbox('', set(filteredColumns))

      summary, results = rp.ttest(data_basic[select1][data_basic['target']==0], data_basic[select1][data_basic['target']==1])
      st.write(summary)

with tab2:
   col21, col22 = st.columns(2)
   with col21:
      st.write(texts['boxplotAbstract'])
   with col22:
      i = 0
      fig, axs = plt.subplots(math.ceil(len(filteredColumns) / 2), 2, figsize=(10, len(filteredColumns)*2))
      for col in filteredColumns:
         axs[math.floor(i/2), i % 2].boxplot(data_basic[col])
         axs[math.floor(i / 2), i % 2].set_title(col)
         i+=1
      st.pyplot(fig)

with tab3:
   st.write(texts['scatterCorrAbstract'])
   numOfScatter = int(len(filteredColumns)*(len(filteredColumns)-1)/2)
   fig, axs = plt.subplots(math.ceil(numOfScatter/2), 2, figsize=(10, numOfScatter*2.5))
   k = 0
   for i in range(0, len(filteredColumns)):
      for j in range(i+1, len(filteredColumns)):
         corr = np.corrcoef(data_basic[filteredColumns[i]], data_basic[filteredColumns[j]])
         axs[math.floor(k / 2), k % 2].set_title(corr[0,1])
         axs[math.floor(k / 2), k % 2].scatter(data_basic[filteredColumns[i]], data_basic[filteredColumns[j]])
         axs[math.floor(k / 2), k % 2].set_xlabel(filteredColumns[i])
         axs[math.floor(k / 2), k % 2].set_ylabel(filteredColumns[j])
         k += 1
   st.pyplot(fig)

with tab4:
   st.write(texts['PCAAbstract'])
   numOfComp = st.slider('Set number of components:', 1, 20, 2)
   pca = PCA(n_components=numOfComp)
   transformed_data = pca.fit_transform(data_basic)
   explained_variance_ratio = pca.explained_variance_ratio_
   st.write(explained_variance_ratio)

with tab5:
   st.write("To add")
