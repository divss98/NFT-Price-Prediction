import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import datetime
import keras

# scaler=StandardScaler()
# n_input=7
# n_features=1
# model=keras.models.load_model("lstm_model")
test_predictions = []


print('hello hi')
dates=[]
datelist = pd.date_range(datetime.datetime.today(), periods=30).tolist()
for i in datelist:
    dates.append(str(i.date()))

#st.title("NFT Analysis and Prediction")
st.set_page_config(layout='wide',page_title="NFT")
st.markdown("<h1 style='text-align: center; color: black;'>NFT Analysis and Prediction</h1>", unsafe_allow_html=True)

col1, col2,col3= st.columns(3)

with col1:
    st.write('')

with col2:
    st.image('https://cdn.thenewstack.io/media/2022/02/e830b7df-apes_punks_nfts-1024x626.jpg',width=750,use_column_width=True)

with col3:
    st.write('')

@st.experimental_singleton
def read_data():
    data2=pd.read_csv('data2.csv')
    return data2

data2=read_data()
categories=sorted(data2['Category'].unique())

@st.experimental_memo
def read_pred_data(option):
    data_lstm=pd.read_csv('data_lstm.csv')
    data_lstm.set_index('Datetime_updated_seconds',inplace=True)
    data_lstm=data_lstm[data_lstm['Category']==option]
    data_lstm.drop(['Category'],axis=1,inplace=True)
    return data_lstm


@st.experimental_memo
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

@st.experimental_memo
def categories_df(option):
    category_subset=data2[data2['Category']==option]
    return category_subset

@st.experimental_memo
def collections_df(option):
    collection_subset=category_subset[category_subset['Collection']==option]
    return collection_subset

@st.experimental_memo
def ID_df(option):
    x=collection_subset[collection_subset['ID_token']==option]
    image=x['Image_url_1'].values
    image=image[0]
    return image

@st.cache()
def lstm_predictions(data):
    scaler=StandardScaler()
    n_input=7
    n_features=1
    model=keras.models.load_model("lstm_model")
    scaler.fit(data)
    first_eval_batch = scaler.transform(np.array(data[-n_input:]))
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(30):
        current_pred = model.predict(current_batch)[0]# append the prediction into the array
        test_predictions.append(current_pred) # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_forecast = scaler.inverse_transform(test_predictions)
    price_forecast=true_forecast/135
    forecast_df=pd.DataFrame(dates,columns=['Date'])
    forecast_df['Price_ETH']=price_forecast*0.00061
    forecast_df['Price_USD']=price_forecast
    forecast_df.set_index('Date',inplace=True)
    return forecast_df




tab1, tab2 = st.tabs(["Analysis and Prediction", "File Uploader"])

with tab2:
    col1, col2= st.columns(2)

    with col1:
        uploadFile = st.file_uploader(label="Upload NFT image", type=['jpg', 'png'])

    # Checking the Format of the page
        if uploadFile is not None:
            # Perform your Manupilations (In my Case applying Filters)
            img = load_image(uploadFile)
            st.image(img,width=500)
            st.write("Image Uploaded Successfully")
        else:
            st.write("Make sure you image is in JPG/PNG Format.")

    with col2:
        if st.button('Predict NFT price'):
            if 'Art' in uploadFile.name:
                data_image=read_pred_data('Art')
            if 'Games' in uploadFile.name:
                data_image=read_pred_data('Games')
            if 'Collectible' in uploadFile.name:
                data_image=read_pred_data('Collectible')
            with st.spinner('Wait for it...'):
                time.sleep(25)
                st.success('Done!')
            st.write('the 1 month prediction is')
            st.write(lstm_predictions(data_image))



with tab1:
    col1, col2 = st.columns(2)

    with col1:
        option = st.selectbox(
            "Select a category to analyse",
            tuple(categories)
        )
    category_subset=categories_df(option)
    counts=category_subset['Collection'].value_counts().rename_axis('Collection').reset_index(name='counts')
    price_data_cat=category_subset[['Datetime_updated','Price_USD']]
    price_data_cat.set_index('Datetime_updated',inplace=True)

    with col2:
        tab1, tab2 = st.tabs(["Bar Chart", "Line Chart"])
        tab1.subheader("Collection Counts for Category:"+option)
        tab1.bar_chart(counts,x="Collection",y='counts',width=100)

        tab2.subheader("Price Trends for Category:"+option)
        tab2.line_chart(price_data_cat)

    col1,col2=st.columns(2)
    collections=list(category_subset['Collection'].value_counts().index)

    with col1:

        option_coll = st.selectbox(
            "Select a collection to analyse",
            collections
        )

    collection_subset=collections_df(option_coll)
    price_data_coll=collection_subset[['Datetime_updated','Price_USD']]
    price_data_coll.set_index('Datetime_updated',inplace=True)

    with col2:
        st.subheader("Price Trends for Collection:"+option_coll)
        st.line_chart(price_data_coll)

    st.markdown("<h3 style='text-align: center; color: black;'>Price Prediction</h3>", unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)

    ID=list(collection_subset['ID_token'].unique())[:20]

    with col1:
        option_col2 = st.selectbox(
            "Select a NFT Token",
            ID
            # label_visibility=st.session_state.visibility,
            # disabled=st.session_state.disabled,
        )
        with st.spinner('Wait for it...'):
            time.sleep(25)
            st.success('Done!')

    image=ID_df(option_col2)
    with col2:
        st.markdown("<h4 style='text-align: center; color: black;'>NFT Image</h4>", unsafe_allow_html=True)
        try:
            st.image(image,width=400)
        except:
            st.write("No NFT image found")
    #
    # data_lstm=read_pred_data(option)
    # preds=lstm_predictions(data_lstm)
    print('hi hello hi 123')
    with col3:
        if st.button('Predict'):
            with st.spinner('Wait for it...'):
                time.sleep(25)
                st.success('Done!')
            st.write('the 1 month prediction is')
            #st.balloons()
            st.write(lstm_predictions(read_pred_data(option)))
