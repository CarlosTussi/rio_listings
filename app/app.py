import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw


def map():

    st.subheader("1) Click on the location on the map: ")

    marker_location = [-22.970294234, -43.18559545]
    
    m = folium.Map(location= marker_location, zoom_start=15)

    if 'marker_location' in st.session_state:
        marker_location = st.session_state['marker_location']
        folium.Marker(marker_location).add_to(m)
        m.location = marker_location
    else:
        marker_location = marker_location 
        st.session_state['marker_location'] = marker_location
        folium.Marker(marker_location).add_to(m)

    
    click_location = st_folium(m, height=300, width=550)

    if click_location and click_location["last_clicked"]:
        latitude = click_location["last_clicked"]["lat"]
        longitude = click_location["last_clicked"]["lng"]
        st.session_state['marker_location'] = [latitude, longitude]
        #marker_location = [latitude, longitude]
        st.rerun()

    st.caption(f"Location coordinates: {marker_location}")


def has_item():
    
    amenities = [
                ("Parking", "has_parking"), 
                ("Pool", "has_pool"),
                ("Washer","has_washer"),
                ("Dishwasher", "has_dishwasher"),
                ("Ceiling Fan", "has_ceiling_fan"),
                ("Long Term Stay", "has_long_term"),
                ("BBQ Grill", "has_bbq_grill"),
                ("Outdoor Area", "has_outdoor"),
                ("Bathtub", "has_bathtub"),
                ("Hot tub", "has_hot_tub"),
                ("AC", "has_ac"),
                ("Seaview", "has_seaview"),
                ]

    st.subheader("2) Select the presence of any of these amenities: ")

    col1, col2, col3 = st.columns(3)

    with col1:
        for ind in range(0,4):
             st.checkbox(amenities[ind][0], key = amenities[ind][1])

    with col2:
        for ind in range(4,8):
             st.checkbox(amenities[ind][0], key = amenities[ind][1])

    with col3:
        for ind in range(8,12):
             st.checkbox(amenities[ind][0], key = amenities[ind][1])
    


def capacity():
    st.subheader("3) Select the maximum capacities: ")


    col1, col2 = st.columns(2)

    with col1:
        accommodates = st.selectbox(
            "Number of Guests: ",
            ([str(x) for x in range(1,10)] + ["10+"]),
        )
        bathroom = st.selectbox(
            "Number of Bathrooms: ",
            ([str(x) for x in range(0,5)]+ ["5+"]),
        )
        bathroom_shared = st.radio(
            "Is bathroom shared?*",
            ["Yes", "No"],
            index=None,
        )
        st.caption("_*Shared with someone strange to guest's party._")

    with col2:
        beds = st.selectbox(
            "Number of Beds: ",
            ([str(x) for x in range(0,8)] + ["8+"]),
        )
        bedrooms = st.selectbox(
            "Number of Bedrooms: ",
            ([str(x) for x in range(0,5)] + ["5+"]),
        )

       

def property_type():

    st.subheader("4) Select the property type: ")

    property__type = st.radio(
            "Property Type ?",
            ["Entire Property", "Private Room", "Shared Room"],
            index=None,
)

def number_of_nights():

    st.subheader("5) Number of Nights: ")

    number_of_nights = st.number_input(
    "Number of nights aavailable in the year : ", min_value = 1, max_value = 356, value=None, placeholder="Type available nights in a year..."
)

    st.selectbox(
            "Minimum nights: ",
            ([str(x) for x in range(1,7)] + ["7+"]),
        )

def reviews():
     st.subheader("6) Reviews: ")

     numnumber_of_reviews = st.number_input(
    "Number of reviews in the last 12 months : ", min_value = 0, value=None, placeholder="Type number of reviews in the last 12 months..."
)
     
     reviews_per_month = st.number_input(
    "Rate of reviews per month : ", min_value = 0.0, value=None, placeholder="Type the rate of reviews per month..."
)
     
     review_score_location = st.number_input(
    "Review Location Score : ", min_value = 0.0, max_value = 5.0, value=None, placeholder="Type the review score for the location..."
     )


     
def description():
    text_area = st.text_area(
        "Property description",
        placeholder="Property description in english",
        height = 200,
    )


def main():

    st.title("Rio rental price predictor")

    map()
    has_item()
    capacity()
    property_type()
    number_of_nights()
    reviews()
    description()
    

    col1, col2 = st.columns(2)

    with col1:  
        st.button("Predict Rental Price", type="primary", use_container_width=True)
    with col2:
        st.button("Reset", type="secondary")

    
    st.header("Predicted Price")
    st.metric(label = "", value = f"R$0.00")
    


if __name__ == "__main__":
    main()