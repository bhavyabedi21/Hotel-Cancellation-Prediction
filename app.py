
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load model and transformer
with open('final_model.joblib', 'rb') as file:
    model = joblib.load(file)

with open('transformer.joblib', 'rb') as file:
    transformer = joblib.load(file)

# Prediction function
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]
    chance = round(pred * 100, 2)
    if pred > 0.5:
        return f'This booking is more likely to get canceled, chances {chance}%'
    else:
        return f'This booking is less likely to get canceled, chances {chance}%'

# Main app
def main():
    st.title('INN HOTEL GROUP - Booking Cancellation Predictor')

    # User inputs
    lt = st.text_input('Enter the lead time in days')
    price = st.text_input('Enter the price of the room')
    weekn = st.text_input('Enter the number of week nights in stay')
    wknd = st.text_input('Enter the number of weekend nights in stay')

    mkt = 1 if st.selectbox('How the booking was made', ['Online', 'Offline']) == 'Online' else 0
    adult = st.selectbox('How many adults', [1, 2, 3, 4])
    arr_m = st.slider('What is the month of arrival?', min_value=1, max_value=12, step=1)

    day_mapping = {'Mon': 0, 'Tues': 1, 'Wed': 2, 'Thurs': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    arr_w = day_mapping[st.selectbox('What is the arrival day?', list(day_mapping.keys()))]
    dep_w = day_mapping[st.selectbox('What is the departure day?', list(day_mapping.keys()))]

    park = 1 if st.selectbox('Does customer need parking', ['Yes', 'No']) == 'Yes' else 0
    spcl = st.selectbox('Number of special requests made by the customer', [0, 1, 2, 3, 4, 5])

    if st.button('Predict'):
        # Check and convert input types
        if not lt or not price or not weekn or not wknd:
            st.error("Please fill in all numeric fields.")
            return

        try:
            lt_val = float(lt)
            price_val = float(price)
            weekn_val = int(weekn)
            wknd_val = int(wknd)
        except ValueError:
            st.error("Enter valid numbers only for lead time, price, week nights, and weekend nights.")
            return

        tot = weekn_val + wknd_val
        df_input = pd.DataFrame([[lt_val, price_val]], columns=['lead_time', 'price'])  # Adjust column names as per training
        lt_t, price_t = transformer.transform(df_input)[0]

        inp_list = [lt_t, spcl, price_t, adult, wknd_val, park, weekn_val, mkt, arr_m, arr_w, tot, dep_w]

        # Show prediction
        response = prediction(inp_list)
        st.success(response)

if __name__ == '__main__':
    main()
