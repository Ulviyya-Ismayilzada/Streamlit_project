import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import streamlit as st



 

st.set_page_config(page_title="The World of Data Scientist",page_icon='Icon',layout='wide')
st.header("Welcome to the World of Data Scientist",divider='rainbow')

st.sidebar.image('Data-Science.jpeg',caption='Predictions have an expiry date.',use_column_width=True)


Choice=st.sidebar.radio(
    label='Manage pages',options=('Homepage','EDA','Modelling')
    )

if Choice =='Homepage':
    
    st.image('data scientist.jpg',caption='Torture the data, and it will confess to anything – Ronald Coase.',use_column_width=True)
 
    st.subheader('Homepage')
    Data=st.selectbox('Select data:',options=('Water Potability','Loan Prediction'))
    
    if Data == 'Water Potability':
       
        df=pd.read_csv ('water_potability.csv')
            
        st.dataframe(df)
        
        st.subheader('About Dataset')
        st.write('The dataset contains 9 predictive attributes and 1 goal field. The target variable refers to the Water Potability.')
        st.markdown('Dataset Description:')
        st.text('''ph: pH of 1. water (0 to 14).
Hardness: Capacity of water to precipitate soap in mg/L.
Solids: Total dissolved solids in ppm.
Chloramines: Amount of Chloramines in ppm.
Sulfate: Amount of Sulfates dissolved in mg/L.
Conductivity: Electrical conductivity of water in μS/cm.
Organic_carbon: Amount of organic carbon in ppm.
Trihalomethanes: Amount of Trihalomethanes in μg/L.
Turbidity: Measure of light emitting property of water in NTU.
Potability: Indicates if water is safe for human consumption. Potable - 1 and Not potable - 0''')
        
                    
    else:
        df=pd.read_csv('loan_pred.csv')
            
        st.dataframe(df)
        
        st.subheader('About Dataset')
        st.write('The dataset contains 11 predictive attributes and 1 goal field. The target variable refers to the Loan Status.')
        st.markdown('Dataset Description:')
        st.text('''Dream Housing Finance company deals in all home loans.
They have presence across all urban, semi urban and rural areas. 
Customer first apply for home loan after that company validates the customer eligibility for loan.
The company wants to automate the loan eligibility process (real time) 
based on customer detail provided while filling online application form. 
These details are Gender, Marital Status, Education, Number of Dependents, 
Income, Loan Amount, Credit History and others. To automate this process,
they have given a problem to identify the customers segments, those are eligible for
loan amount so that they can specifically target these customers.''' )
     
    st.sidebar.subheader('Mark Twain')
    st.sidebar.markdown("Data is like garbage.You’d better know what you are going to do with it before you collect it.")
    
    st.sidebar.subheader('Jay Baer')
    st.sidebar.markdown('We are surrounded by data, but starved for insights.')      
    
    
elif Choice=='EDA':
    st.subheader('EDA')
    a=st.radio('Dataset:', options=('Water Potability','Loan Prediction'))
    if a == 'Water Potability':
        st.header('EDA for Water Potability dataset')
        st.image('water2.jpg',caption='Water Potability Dataset')
        df=pd.read_csv ('water_potability.csv')
        
        st.subheader('Header of dataset')
        st.write(df.head())
        
        st.subheader('Statistical analysis of dataset')
        st.write(df.describe())
        
         
        col1,col2=st.columns(2)
        with col1:   
            st.subheader('Null Values')
            st.write(df.isna().sum())
        
        with col2:
            
            Imputation=st.selectbox('Imputation', options=('Mean','Median'))
            if Imputation =='Mean':
                 for col in df.columns:
                    df[col]=df[col].fillna(df[col].mean())
                
            else:
                for col in df.columns:
                    df[col]=df[col].fillna(df[col].median())
        
            st.subheader('Null Values')   
            st.write(df.isna().sum()) 
        st.subheader('After imputation dataset')
        st.write(df.head())
        
        
        col1,col2=st.columns(2)
        with col1:
            st.subheader('Box Plot')
                        
            col1_box=px.box(df)
            st.plotly_chart(col1_box)
       
        with col2:
            st.subheader('Scatter Plot')
            col12_scatter=px.scatter(df)
            st.plotly_chart(col12_scatter)
        
        st.subheader('Bar Chart')
        df_bar=px.bar(x=df['Potability'].value_counts().unique(),y=df['Potability'].value_counts())
        st.plotly_chart(df_bar)        
        st.subheader('Dataset after removing outliers')
        Q1=np.percentile(df,25)
        Q3=np.percentile(df,75)
        upperbound=Q3+(1.5*(Q3-Q1))
        lowerbound=Q1-(1.5*(Q3-Q1))
        df=np.clip(df,lowerbound,upperbound)
        st.write(df)
        df.to_csv('new_water_df.csv',index=False)        
                
    else:
        st.header('EDA for Loan Prediction dataset')
        st.image('loanpredict.jpg',caption='Loan Prediction Dataset')
        df=pd.read_csv ('loan_pred.csv')
        
        st.subheader('Header of dataset')
        st.write(df.head())
        
        st.subheader('Statistical analysis of dataset')
        st.write(df.describe())
        
         
        col1,col2=st.columns(2)
        with col1:   
            st.subheader('Null Values')
            st.write(df.isna().sum())
        
        with col2:
            
            Imputation=st.selectbox('Imputation', options=('Mode','Backfill'))
            if Imputation =='Mode':
                 for col in df.columns:
                    df[col]=df[col].fillna(df[col].mode().iloc[0])
                
            else:
                df=df.fillna(method='bfill')
        
            st.subheader('Null Values')   
            st.write(df.isna().sum()) 
        st.subheader('After imputation of dataset')
        st.write(df.head())
        
        col1,col2=st.columns(2)
        with col1:
            st.subheader('Box Plot')
            col1_box=px.box(df.select_dtypes(exclude='object'))
            st.plotly_chart(col1_box)
       
        with col2:
            st.subheader('Scatter Plot')
            col12_scatter=px.scatter(df.select_dtypes(exclude='object'))
            st.plotly_chart(col12_scatter)
         
        st.subheader('Bar Chart')
        df_bar=px.bar(x=df['Loan_Status'].value_counts().unique(),y=df['Loan_Status'].value_counts())
        st.plotly_chart(df_bar)        
        st.subheader('Dataset after removing outliers')
        def remove_outliers(col):
            Q1=np.percentile(col,25)
            Q3=np.percentile(col,75)
            upperbound=Q3+(1.5*(Q3-Q1))
            lowerbound=Q1-(1.5*(Q3-Q1))
            return np.clip(col,lowerbound,upperbound)
        num_df=df.select_dtypes(exclude='object').columns
        df[num_df]=df[num_df].apply(remove_outliers)
        st.write(df)
        df.to_csv('new_loan_df.csv',index=False)
                                           
    
    
    st.sidebar.subheader('Dan Heath')
    st.sidebar.markdown("Data are just summaries of thousands of stories—tell a few of those stories to help make the data meaningful.")
    
    st.sidebar.subheader('Dean Abbott')
    st.sidebar.markdown('No data is clean, but most is useful.')
       
    
    
else:
    st.subheader('Modelling')
    c=st.radio('Dataset:', options=('Water Potability','Loan Prediction'))
    if c=='Water Potability':
        df=pd.read_csv('new_water_df.csv')
        st.dataframe(df)
        X=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        scale=st.selectbox("Scaling",options=('Standard Scaler','MinMax Scaler','Robust Scaler'))
        from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
        if scale=='Standard Scaler':
            X=StandardScaler().fit_transform(X)
        elif scale=='MinMax Scaler':
            df=MinMaxScaler().fit_transform(df)
        else:
            df=RobustScaler().fit_transform(df)
        
        
    else:
        df=pd.read_csv('new_loan_df.csv')
        st.dataframe(df)
        col1,col2=st.columns(2)
        with col1:
            encode=st.radio('Encoding',options=('Ordinal Encoder','One Hot Encoder'))
            from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
            df_cat=df.select_dtypes(include='object')
            df_num=df.select_dtypes(exclude='object')
            if encode =='Ordinal Encoder':
                df_cat =pd.DataFrame(OrdinalEncoder().fit_transform(df_cat),columns=df_cat.columns)
            else:
                df_cat=pd.get_dummies(df_cat,drop_first=True,dtype=int)
                #df_cat =OneHotEncoder(drop='first').fit_transform(df,columns=df_cat)
            df=pd.concat([df_cat,df_num],axis=1)
            st.write(df)
        with col2:
            scale=st.selectbox("Scaling",options=('Standard Scaler','MinMax Scaler','Robust Scaler'))
            from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
            X=df.iloc[:,:-1]
            y=df.iloc[:,-1]
            if scale=='Standard Scaler':
                X=StandardScaler().fit_transform(X)
            elif scale=='MinMax Scaler':
                df=MinMaxScaler().fit_transform(df)
            else:
                df=RobustScaler().fit_transform(df)


    from sklearn.model_selection import train_test_split
    df=pd.DataFrame(df)
        
      
    
       
    col1,col2=st.columns(2)
    with col1:
        r_state=st.slider('Random state:',min_value=1,max_value=50, step=3)     
        
    
    with col2:
        t_size=st.slider('Test size:', min_value=0.2,max_value=0.4)      
     
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=t_size,random_state=r_state, stratify=y)
    
    st.subheader('Train and Testing')
    models=st.radio('Models:',options=('XGBoost','Catboost'))
    if models=='XGBoost':
        xgbt=XGBClassifier()
        xgbt.fit(X_train,y_train)
        y_pred=xgbt.predict(X_test)
    else:
        catbc=CatBoostClassifier()
        catbc.fit(X_train,y_train)
        y_pred=catbc.predict(X_test)
    from sklearn.metrics import accuracy_score,confusion_matrix
    st.write('Accuracy score:',round(accuracy_score(y_test,y_pred)*100,2))
    
    st.markdown('Confusion Matrix')
    st.write('Confusion matrix:', confusion_matrix(y_test,y_pred))
    
    
    
    st.sidebar.subheader('Albert Einstein')
    st.sidebar.markdown("If you can’t explain it simply, you don’t understand it well enough")
    
    st.sidebar.subheader('Alexander Peiniger')
    st.sidebar.markdown('In the end you should only measure and look at the numbers that drive action, meaning that the data tells you what you should do next.')
    
    st.title('Thanks for attention')
    st.markdown('Author Ms.Ulviyya Ismayilzadeh')