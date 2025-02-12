'''
    User interface components
'''


import streamlit as st
import pandas as pd
import numpy as np

import folium
from streamlit_folium import st_folium

from source.app.processing_predict import process_and_predict
from source.config import *


import regex as re

'''
    def map_component()
    -------------------

    - Allows user to click on the map, capturing the coordinates of that click.

    - features:
        - longitude
        - latitude

'''
def map_component():

    st.subheader("Indicate property location on the map", divider="blue")

    marker_location = RIO_COORDINATES 
    
    # Default value
    if("longitude" not in st.session_state["model_input"]):
        st.session_state["model_input"].update({
                "longitude": RIO_COORDINATES[1],
                "latitude":  RIO_COORDINATES[0]
            })
    
    m = folium.Map(location= marker_location, zoom_start=15)

    # If state variable already exists, updates current marker location
    if 'marker_location' in st.session_state:
        marker_location = st.session_state['marker_location']
        folium.Marker(marker_location).add_to(m)
        m.location = marker_location
    #If state variable does not exists, create it and update marker location
    else:
        st.session_state['marker_location'] = marker_location
        folium.Marker(marker_location).add_to(m)

    # Retrieves user lick location
    click_location = st_folium(m, height=300, width=550)

    # Updated coordinates of user's click location
    if click_location and click_location["last_clicked"]:
        latitude = click_location["last_clicked"]["lat"]
        longitude = click_location["last_clicked"]["lng"]
        st.session_state['marker_location'] = [latitude, longitude]

        #Update longitude and latitude
        if "model_input" in st.session_state:
            st.session_state["model_input"].update({
                "longitude": longitude,
                "latitude": latitude
            })
        # Force the display of the updated marker
        st.rerun()

    st.caption(f"Location coordinates: {marker_location}")

'''
    def has_item_component()
    ------------------------

    - Allows user to select the presence of predefined amenities.

    - features:
        has_pool
        has_bathtub
        ...

'''
def has_item_component():
    
    amenities = AMENITIES_GUI

    st.subheader("Select if amenities available ", divider="blue")

    # Divide the amenitites in 3 columns for beetter visualisation
    col1, col2, col3 = st.columns(3)
    chk_var = {}

    with col1:
        for ind in range(0,4):
             chk_var.update(
                 {
                  amenities[ind][1] : st.checkbox(amenities[ind][0])
                  })

    with col2:
        for ind in range(4,8):
             chk_var.update(
                 {
                  amenities[ind][1] : st.checkbox(amenities[ind][0])
                  })

    with col3:
        for ind in range(8,12):
             chk_var.update(
                 {
                  amenities[ind][1] : st.checkbox(amenities[ind][0])
                  })

    # Update the model input data dictionary with the amenitites selected by the user
    for feature, var in chk_var.items():
        is_checked = 1 if var else 0
        st.session_state["model_input"].update({feature:is_checked})
    

'''
    def update_total_value(feature_name, value, type = "int")
    ---------------------------------------------------------

    - This function updates the 'model_input' session state variable with values input from the user.

    - Input:
      *****
        - feature_name: str
        - value: numeric
        - type: str

    - Output:
      *******
        - None


'''
def update_total_value(feature_name, value, type = "int"):

    # Detect the "+" character for (10+, 5+, etc...)
    plus_pattern = ".*\+"

    if(type == "int"):
        if(value != ""):
            total = 0
            if(re.match(plus_pattern, str(value))):
                total = int(value[:-1])
            else:
                total = int(value)

            st.session_state["model_input"].update({feature_name: total})
    elif(type == "float"):
        st.session_state["model_input"].update({feature_name: value})


'''
    def capacity_component()
    ------------------------

    - Alows the user to input the capacity values for some features.

    - features:
        - accommodates
        - bathrooms
        - is_bathroom_shared
        - beds
        - bedrooms
        
'''
def capacity_component():
    st.subheader("Select the maximum capacities ", divider="blue")


    col1, col2 = st.columns(2)

    with col1:
        accommodates = st.selectbox(
            "Number of Guests: ",
            ([str(x) for x in range(1,ACCOMM_LIM_FT)] + [str(ACCOMM_LIM_FT)+"+"]),
        )
        update_total_value("accommodates", accommodates)
        
        bathrooms = st.selectbox(
            "Number of Bathrooms: ",
            ([str(x) for x in range(0,BATHROOM_LIM_FT)]+ [str(BATHROOM_LIM_FT)+"+"]),
        )
        update_total_value("bathrooms", bathrooms)


        bathroom_shared = st.radio(
            "Is bathroom shared?*",
            ["No", "Yes"],
        )
        
        # Will force the the first element to be pre-selected
        if bathroom_shared == "No":
            pass

        st.session_state["model_input"].update({"is_bathroom_shared" : 1 if bathroom_shared == "Yes" else 0})

        st.caption("_*Shared with someone strange to guest's party._")

    with col2:
        beds = st.selectbox(
            "Number of Beds: ",
            ([str(x) for x in range(0,BEDS_LIM_FT)] + [str(BEDS_LIM_FT) + "+"]),
        )
        update_total_value("beds", beds)

        bedrooms = st.selectbox(
            "Number of Bedrooms: ",
            ([str(x) for x in range(0,BEDROOMS_LIM_FT)] + [str(BEDROOMS_LIM_FT)+"+"]),
        )
        update_total_value("bedrooms", bedrooms)

       
'''
    def property_type_component()
    -----------------------------

    - Radio option button to select the property type.

    - feature:
        - property_type

'''
def property_type_component():

    st.subheader("Select the property type ", divider="blue")

    property_type = st.radio(
            "Property Type ?",
            ["Entire Property", "Private Room", "Shared Room"],
)
    
    # Will force the the first element to be pre-selected
    if property_type == "Entire Property":
        pass

    st.session_state["model_input"].update({

                    "room_type_Entire home/apt" : 1 if property_type == "Entire Property" else 0,
                    "room_type_Private room" :  1 if property_type == "Private Room" else 0,
                    "room_type_Shared room":  1 if property_type == "Shared Room" else 0,
    })

'''
    def number_of_nights_component()
    --------------------------------

    - Alows the user to indicate numeric values for features related to number of nights.

    -features:
        - availability_365
        - minimum_nights_avg_ntm

'''
def number_of_nights_component():

    st.subheader("Define the following number of nights ", divider="blue")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        availability_60 = st.number_input(
            "Number of nights available in the next 60 days : ", min_value = 0, max_value = 30, value=1, placeholder="Type available nights in 60 days..."
        )
        update_total_value("availability_60", availability_60)
    
    with col2:
        availability_90 = st.number_input(
            "Number of nights available in the next 90 days : ", min_value = 0, max_value = 90, value=1, placeholder="Type available nights in 90 days ..."
        )
        update_total_value("availability_90", availability_90)

    with col3:
        availability_365 = st.number_input(
            "Number of nights available in the next 365 days : ", min_value = 0, max_value = 356, value=1, placeholder="Type available nights in the next year..."
        )
        update_total_value("availability_365", availability_365)

    minimum_nights_avg_ntm = st.selectbox(
            "Minimum nights per stay: ",
            ([str(x) for x in range(1,MIN_NIGHTS_LIM_FT)] +[str(MIN_NIGHTS_LIM_FT)+"+"]),
        )
    update_total_value("minimum_nights_avg_ntm", minimum_nights_avg_ntm)

'''
    def reviews_component()
    -----------------------

    -  Inteface components to capture inforamtion about reviews.

    - features:
        -numnumber_of_reviews_ltm
        -reviews_per_month
        -review_scores_location

'''
def reviews_component():
    st.subheader("Define the following reviews' information ", divider="blue")

    numnumber_of_reviews_ltm = st.number_input(
    "Number of reviews in the last 12 months : ", min_value = 0, value=0, placeholder="Type the number of reviews in the last 12 months..."
)
    update_total_value("number_of_reviews_ltm", numnumber_of_reviews_ltm)
     
    reviews_per_month = st.number_input(
    "Rate of reviews per month : ", min_value = 0.0, value=0.0, placeholder="Type the rate of reviews per month..."
)
    update_total_value("reviews_per_month", reviews_per_month, "float")
     
    review_scores_value = st.number_input(
    "Review Value Score : ", min_value = 0.0, max_value = 5.0, value=0.0, placeholder="Type the review score for property value..."
     )

    update_total_value("review_scores_value", review_scores_value, "float")
     
    # Feature (flag) indicating absence of reviews
    if(review_scores_value == 0):
        st.session_state["model_input"].update({"is_score_empty": 1})


'''
    def description_component()
    ---------------------------

    - Text area that receives the property description from the user.

    -features:
        - description

'''
def description_component():
    st.subheader("Property description", divider="blue")

    text_area = st.text_area(
        "Describe your property",
        placeholder="Property description in english",
        height = 200,
    )

    st.session_state["model_input"].update(
        {"description": text_area}
    )


'''
    def buttons_component(price_pred_model, geo_cluster_pred_model, scaler_transf_model, df_model_input)
    ------------------------------------------------------------------------------------------------------

    - This components initiates the data transformation and price prediction.
    - It also displays the price preddicted.

    - Input:
      ******
        - price_pred_model : HistGradientBoostingRegressor()    # Trained model
        - geo_cluster_pred_model : KMeans()                     # Trained model
        - scaler_transf_model : MaxMinScaler()                  # Trained model
        - df_model_input : pd.DataFrame

    - feature:
        - price


'''
def buttons_component(price_pred_model, geo_cluster_pred_model, scaler_transf_model, df_model_input):

    # Buttons to run the model or reset
    col1, col2 = st.columns(2)
    predicted_price = 0.0

    # Predict final price
    if col2.button("Predict Rental Price", type="primary", use_container_width=True):
        
        ###################
        # Preprocess Data #
        ###################
        predicted_price = process_and_predict(df_model_input, price_pred_model, geo_cluster_pred_model, scaler_transf_model)
    

    #####################
    # Display the price #
    #####################
    st.header("")
    container = st.container(border=True, key = "price-container")
    container.header("Predicted Price", divider="red")
    container.metric(label = " ", value = f"R${predicted_price:.2f}")

    # Adding custom CSS just to better align the price in the container
    st.markdown("""
                <style>
                    .st-key-price-container{
                        text-align: center;
                    }
                </style>
                """, unsafe_allow_html=True)
    st.header("")
'''
    def df_component(df_display)
    ----------------------------

    - Displays a dataframe with the data input byt the user.

    - Input:
      ******
      - df:display : pd.DataFrame

'''
def df_component(df_display):

    st.subheader("Input Data", divider="grey")
    # Display the dataframe used in the model
    col1, col2, col3 = st.columns([1.5,2,2])


    with col1:
        st.dataframe(df_display.iloc[0:10, :])
    with col2:
        st.dataframe(df_display.iloc[10:20, :], use_container_width = True)
    with col3:
        st.dataframe(df_display.iloc[20:, :], use_container_width = True)

'''
    def gui(price_pred_model, geo_cluster_pred_model, scaler_transf_model)

    - Centralizes all the componentes in order of appearance for the user.

    - Input:
      *****
        - price_pred_model : HistGradientBoostingRegressor()    # Trained model
        - geo_cluster_pred_model : KMeans()                     # Trained model
        - scaler_transf_model : MaxMinScaler()                  # Trained model   
        - price_model_name                                      # Name of the price model being used
        - price_model_last_modified                             # Date of last model update


'''
def gui(price_pred_model, geo_cluster_pred_model, scaler_transf_model, price_model_name, price_model_last_modified):

    st.set_page_config(page_title="Rio Rental Predictor")
    #################
    # PROJECT TITLE #
    #################
    st.title("Rio Rental Predictor")

    # State variable that will contain user's input
    model_input = {}
    if "model_input" not in st.session_state:
        st.session_state["model_input"] = model_input

    #######################
    # INFORMATION SIDEBAR #
    #######################
    with st.sidebar:
        st.title("Rio de Janeiro, Brazil")
        st.write("Rental Predictor")
        st.divider()
        st.header("Model Used")
        st.caption(f"* _{price_model_name}_")
        st.header("Model Last Modified")
        st.caption(f"* _{price_model_last_modified}_")
        st.divider()
        st.header("About")
        st.write("* For project details click [here](%s)" % "https://github.com/CarlosTussi/rio_listings")
        st.write("* Please check the license [here](%s)" % "https://github.com/CarlosTussi/rio_listings?tab=MIT-1-ov-file")
        
        

    #############################
    # MAIN INTERFACE COMPONENTS #
    #############################
    map_component()
    has_item_component()
    capacity_component()
    property_type_component()
    number_of_nights_component()
    reviews_component()
    description_component()

    # Model Input as a Data Frame
    df_model_input = pd.DataFrame(st.session_state["model_input"], index = ["features"])
    buttons_component(price_pred_model, geo_cluster_pred_model, scaler_transf_model, df_model_input)


    # Data frame that will be display to the user before its transformations
    df_display = df_model_input.copy().T
    df_component(df_display)