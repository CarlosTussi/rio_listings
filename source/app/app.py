
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

from pipelines import pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import regex as re
    
def map():

    st.subheader("1) Click on the location on the map: ")

    marker_location = [-22.970294234, -43.18559545] #Rio
    
    # Default value
    if("longitude" not in st.session_state["model_input"]):
        st.session_state["model_input"].update({
                "longitude": -43.18559545,
                "latitude": -22.970294234
            })
    
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

        #Update longitude and latitude
        if "model_input" in st.session_state:
            st.session_state["model_input"].update({
                "longitude": longitude,
                "latitude": latitude
            })
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

    for feature, var in chk_var.items():
        is_checked = 1 if var else 0
        st.session_state["model_input"].update({feature:is_checked})
    


def is_Checked():
    pass


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



def capacity():
    st.subheader("3) Select the maximum capacities: ")


    col1, col2 = st.columns(2)

    with col1:
        accommodates = st.selectbox(
            "Number of Guests: ",
            ([str(x) for x in range(1,10)] + ["10+"]),
        )
        update_total_value("accommodates", accommodates)
        
        bathrooms = st.selectbox(
            "Number of Bathrooms: ",
            ([str(x) for x in range(0,5)]+ ["5+"]),
        )
        update_total_value("bathrooms", bathrooms)


        bathroom_shared = st.radio(
            "Is bathroom shared?*",
            ["Yes", "No"],
            index=None,
        )

        st.session_state["model_input"].update({"is_bathroom_shared" : 1 if bathroom_shared == "Yes" else 0})

        st.caption("_*Shared with someone strange to guest's party._")

    with col2:
        beds = st.selectbox(
            "Number of Beds: ",
            ([str(x) for x in range(0,8)] + ["8+"]),
        )
        update_total_value("beds", beds)

        bedrooms = st.selectbox(
            "Number of Bedrooms: ",
            ([str(x) for x in range(0,5)] + ["5+"]),
        )
        update_total_value("bedrooms", bedrooms)

       

def property_type():

    st.subheader("4) Select the property type: ")

    property_type = st.radio(
            "Property Type ?",
            ["Entire Property", "Private Room", "Shared Room"],
            index=None,
)

    st.session_state["model_input"].update({

                    "room_type_Entire home/apt" : 1 if property_type == "Entire Property" else 0,
                    "room_type_Private room" :  1 if property_type == "Private Room" else 0,
                    "room_type_Shared room":  1 if property_type == "Shared Room" else 0,
    })


def number_of_nights():

    st.subheader("5) Number of Nights: ")

    availability_365 = st.number_input(
    "Number of nights available in the year : ", min_value = 1, max_value = 356, value=1, placeholder="Type available nights in a year..."
)
    update_total_value("availability_365", availability_365)

    minimum_nights_avg_ntm = st.selectbox(
            "Minimum nights: ",
            ([str(x) for x in range(1,7)] + ["7+"]),
        )
    update_total_value("minimum_nights_avg_ntm", minimum_nights_avg_ntm)

def reviews():
    st.subheader("6) Reviews: ")

    numnumber_of_reviews_ltm = st.number_input(
    "Number of reviews in the last 12 months : ", min_value = 0, value=0, placeholder="Type the number of reviews in the last 12 months..."
)
    update_total_value("number_of_reviews_ltm", numnumber_of_reviews_ltm)
     
    reviews_per_month = st.number_input(
    "Rate of reviews per month : ", min_value = 0.0, value=0.0, placeholder="Type the rate of reviews per month..."
)
    update_total_value("reviews_per_month", reviews_per_month, "float")
     
    review_scores_location = st.number_input(
    "Review Location Score : ", min_value = 0.0, max_value = 5.0, value=0.0, placeholder="Type the review score for the location..."
     )

    update_total_value("review_scores_location", review_scores_location, "float")
     
    # Feature (flag) indicating absence of reviews
    if(review_scores_location == 0):
        st.session_state["model_input"].update({"is_score_empty": 1})


     
def description():
    text_area = st.text_area(
        "Property description",
        placeholder="Property description in english",
        height = 200,
    )

    st.session_state["model_input"].update(
        {"description": text_area}
    )


def main(model, geo_cluster_model, normaliser_model):

    st.title("Rio rental price predictor")

    model_input = {}

    if "model_input" not in st.session_state:
        st.session_state["model_input"] = model_input

    map()

    if "model_input" in st.session_state:
        print(st.session_state["model_input"])

    has_item()
    capacity()
    property_type()
    number_of_nights()
    reviews()
    description()
    
    # Model Input
    df_model_input = pd.DataFrame(st.session_state["model_input"], index = ["features"])
    predicted_price = 0.0

    df_display = df_model_input.copy().T

    # Buttons to run the model or reset
    col1, col2 = st.columns(2)

    # Predict final price
    if col1.button("Predict Rental Price", type="primary", use_container_width=True):
        # Pre-process data pipeline
        preprocess_data = Pipeline([
            ('cat_feat_extraction_pipeline_app', pipelines.cat_feat_extraction_pipeline_app),  
        ])

        # Predict geo-clusyter model

        cluster = geo_cluster_model.predict(df_model_input.loc[:,["latitude", "longitude"]])[0]
        df_model_input["geo_cluster"] = cluster

        # One hot-encode representation (manually encoding all 25 categories) and indicate the cluster which current geo-cluster number belongs to
        for i in range(0,25):
            df_model_input["geo_cluster_"+str(i)] = 0
        df_model_input["geo_cluster_"+str(cluster)] = 1
        df_model_input = df_model_input.drop(["latitude", "longitude", "geo_cluster"], axis = 1)

        X = preprocess_data.fit_transform(df_model_input)


        # Reordering to transofrm with the normaliser (MaxMin required the same order)
        X_normalise = pd.DataFrame()
        for col in normaliser_model.feature_names_in_:
            X_normalise[col] = X[col]

        X = normaliser_model.transform(X_normalise)


        # Model prediction
        predicted_price = model.predict(X)

        # Display the price
        st.header("Predicted Price")
        st.metric(label = " ", value = f"R${predicted_price[0]:.2f}")
    
    # Reset all input
    if col2.button("Reset", type="secondary"):
        # TO-DO
        pass



    # Display the dataframe used in the model
    col1, col2, col3 = st.columns(3)


    with col1:
        st.dataframe(df_display.iloc[0:9, :])
    with col2:
        st.dataframe(df_display.iloc[9:19, :])
    with col3:
        st.dataframe(df_display.iloc[19:, :])

    #print(df_model_input)
    


if __name__ == "__main__":
    
    model = joblib.load("../models/price_model.joblib")
    geo_cluster_model = joblib.load("../models/geo_kmeans_model.joblib")
    normaliser_model = joblib.load("../models/normaliser_model.joblib")

    main(model, geo_cluster_model, normaliser_model)