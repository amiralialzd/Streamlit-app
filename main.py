import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import plotly.figure_factory as ff
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
import numpy as np


df=pd.read_csv('spacex_dataset_part_2.csv')
print("rows, cols:", df.shape)
print("success rate:", df['Class'].mean())
print("payload min/max:", df['PayloadMass'].min(), df['PayloadMass'].max())
print(df['LaunchSite'].value_counts())
add_launchsite = st.sidebar.selectbox(
    "choose your launch site",
    ("CCAFS SLC 40", "KSC LC 39A",'VAFB SLC 4E','ALL'),index=3
)
st.caption("Note: the model is trained on the full dataset; the charts above reflect your filter, but the model evaluation does not.")
st.dataframe(df)

X=df.drop(columns=['Class','Date','BoosterVersion','Orbit','LaunchSite','Legs','GridFins','Reused','LandingPad','Outcome','Serial'])
Y=df['Class'].to_numpy()
transform=preprocessing.StandardScaler()
X=transform.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)




with st.sidebar:
    add_model = st.radio(
        "Choose a method",
        ("DecisionTreeClassifier", "KNeighborsClassifier"))

start_num , end_num =st.select_slider('select a range of numbers',options=range(0,13621),value=(0,13620))

ranges=[start_num, end_num]


st.write(f'here you can see the chosen model {add_model} and chosen launch site { add_launchsite} '
         f'and payload mass range {start_num,end_num}')





def get_inputs(add_launchsite, add_model,ranges):
    payload_filtered = df[
        (df['PayloadMass'] >= ranges[0]) &
        (df['PayloadMass'] <= ranges[1])
        ]
    if add_launchsite !='ALL':
        filtered_df=payload_filtered[payload_filtered['LaunchSite']==add_launchsite]
        success_rate = filtered_df['Class'].mean()
        rate_df = pd.DataFrame({
            'Result': ['Success', 'Failure'],
            'Rate': [success_rate, 1 - success_rate]
        })
        fig = px.pie(rate_df, names='Result', values='Rate', title='Success Rate')
        scatter_fig = px.scatter(filtered_df, x='PayloadMass', y='Class', color='LaunchSite')



    else:
        filtered_df=payload_filtered
        success_rate = filtered_df['Class'].mean()
        rate_df = pd.DataFrame({
            'Result': ['Success', 'Failure'],
            'Rate': [success_rate, 1 - success_rate]
        })

        fig=px.pie(rate_df,names='Result',values='Rate',title='Success rate')
        scatter_fig = px.scatter(filtered_df, x='PayloadMass', y='Class', color='LaunchSite')
    st.plotly_chart(fig)
    st.plotly_chart(scatter_fig)

    if add_model=='DecisionTreeClassifier':
        parameters = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [2 * n for n in range(1, 10)],
                      'max_features': ['sqrt', 'log2'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10]}

        Tree=DecisionTreeClassifier()
        tree_cv=GridSearchCV(Tree,parameters,cv=10)
        tree_cv.fit(X_train,Y_train)
        yhat = tree_cv.predict(X_test)
        cm=confusion_matrix(Y_test,yhat)

        fig = ff.create_annotated_heatmap(
          cm.tolist(),
          x=['Predicted 0', 'Predicted 1'],
          y=['Actual 0', 'Actual 1'],
          colorscale='Blues' )
    else:
        parameters = {'n_neighbors': list(range(1, 11)),
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1, 2]}

        KNN = KNeighborsClassifier()
        knn_cv = GridSearchCV(KNN, parameters, cv=10)
        knn_cv.fit(X_train, Y_train)
        yhat = knn_cv.predict(X_test)
        cm=confusion_matrix(Y_test,yhat)
        fig = ff.create_annotated_heatmap(
          cm.tolist(),
          x=['Predicted 0', 'Predicted 1'],
          y=['Actual 0', 'Actual 1'],
          colorscale='Blues' )


    st.plotly_chart(fig)








get_inputs(add_launchsite, add_model,ranges)