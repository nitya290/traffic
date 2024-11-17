
# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# Set up the app title and image
st.title('Traffic Volume Predictor ')
st.image('traffic_image.gif', use_column_width = True, 
         caption = "Utilize our advanced machine learning application to predict traffic volume")


# Reading the pickle file that we created before 
model_pickle = open('reg_traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

default_df = pd.read_csv('Traffic_Volume.csv')
default_df['date_time'] = pd.to_datetime(default_df['date_time'])
default_df['month'] = default_df['date_time'].dt.month
default_df['weekday'] = default_df['date_time'].dt.day_of_week
default_df['hour'] = default_df['date_time'].dt.hour
default_df = default_df.drop('date_time', axis=1)

sample_df = default_df.drop(columns=['traffic_volume'])
st.sidebar.image('traffic_sidebar.jpg')

with st.sidebar.expander("Option 1: Upload CSV"):
    st.header('Upload a CSV file containing the diamond details')
    userdiamond = st.file_uploader('Choose a CSV File')
    st.header('Sample Data Format for Upload')
    st.write(sample_df.head())

with st.sidebar.expander("Option 2: Fill Out Form"):
    with st.form("Option 2: Fill Out Form"):

        st.header("Diamond Features Input")
        
        holiday = st.selectbox('Holiday', options=['None', 'Columbus Day','Independence Day','Christmas Day','Labor Day','Martin Luther King Jr Day','Memorial Day','New Years Day','State Fair','Thanksgiving Day','Veterans Day','Washingtons Birthday'])
        temp = st.number_input('Temperature', min_value=default_df['temp'].min(), max_value=default_df['temp'].max(), step=1.0)
        rain = st.number_input('Rain (mm)', min_value=default_df['rain_1h'].min(), max_value=default_df['rain_1h'].max(), step=1.0)
        snow = st.number_input('Snow', min_value=default_df['snow_1h'].min(), max_value=default_df['snow_1h'].max(), step=1.0)
        clouds = st.number_input('Clouds', min_value=default_df['clouds_all'].min(), max_value=default_df['clouds_all'].max(), step=1)
        weather = st.selectbox('Weather', options= default_df['weather_main'].unique())
        #year = st.selectbox('Year', options = default_df['year'].unique())
        month = st.selectbox('Month', options = default_df['month'].unique())
        day = st.selectbox('Weekday', options = default_df['weekday'].unique())
        hour = st.selectbox('Hour', options = default_df['hour'].unique())
        submit_button = st.form_submit_button("Predict")
alpha = st.slider('Select alpha value for prediction intervals', min_value=0.1, max_value=0.50, value =0.1, step=0.01) 
confidence = (1- alpha)*100
is_holiday = 0 if holiday == 'None' else 1



     # Encode the inputs for model prediction
    

encode_df = default_df.copy()
encode_df = encode_df.drop(columns=['traffic_volume'])
encode_df['holiday'] = encode_df['holiday'].fillna('None').apply(lambda x: 0 if x == 'None' else 1)

# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [is_holiday, temp, rain, snow, clouds, weather, month, day, hour]

# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df)

#     # Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

#     # Get the prediction with its intervals

alpha = alpha 
prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
for i in range (len(user_encoded_df)):
    pred_value = prediction[0]
    lower_limit = intervals[:, 0][0][0]
    upper_limit = intervals[:, 1][0][0]





# Show the prediction on the app


# Display results using metric card

st.metric(label = "Predicted Volume", value = f"${pred_value :.2f}")

st.write(f"**Confidence Interval**: [{lower_limit :.2f}, {upper_limit :.2f}]")

    



if userdiamond:
    user_df = pd.read_csv(userdiamond)  # User-provided data
    
    # Prepare the original dataframe
    original_df = default_df.drop(columns=['traffic_volume'])
    original_df['holiday'] = original_df['holiday'].fillna('None').apply(lambda x: 0 if x == 'None' else 1)

    # Ensure the order of columns in user data matches the original data
    user_df = user_df[original_df.columns]

    # Combine the original and user dataframes along rows (axis=0)
    combined_df = pd.concat([original_df, user_df], axis=0)

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)
##############################################################################################################
#Used ChatGPT to debug. See appendix for further details
    combined_df_encoded = combined_df_encoded.reindex(columns=pd.get_dummies(original_df).columns, fill_value=0)
    original_rows = original_df.shape[0]
    original_df_encoded = combined_df_encoded.iloc[:original_rows]
    user_df_encoded = combined_df_encoded.iloc[original_rows:]

  
    user_df_encoded = user_df_encoded.astype(original_df_encoded.dtypes)
#############################################################################
    # Predictions for user data
    user_pred = reg_model.predict(user_df_encoded)

    # Predicted species
    user_pred_species = user_pred

    # Adding predicted species to user dataframe
    user_df['Predicted Volume'] = user_pred_species

    # # Get the prediction with its intervals
    # lower_limit1=[]
    lower_limits=[]
    upper_limits=[]
    
    alpha = alpha 
    prediction1, intervals1 = reg_model.predict(user_df_encoded, alpha = alpha)
    for i in range(len(user_df_encoded)):
    
        lower_limits.append(intervals1[i, 0]) 
        upper_limits.append(intervals1[i, 1])  


    user_df['CI Lower'] = lower_limits
    user_df['CI Upper'] = upper_limits


    st.subheader("Predicting Diamond Prices")
    st.dataframe(user_df)

st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")