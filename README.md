Airbnb Propeprty Price Predictor 

## Table of Contents
- [About](#about)
- [Data](#data)
    - [Source](#source)
    - [Overview](#overview)
        - [Data Dictionary](#data-dictionary)
        - [Features](#features)
- [Methodology](#methodology)
    - [Data Cleaning](#data-cleaning)
    - [EDA and FE](#eda-and-fe)
    - [Feature Selection](#feature-selection)
    - [Model](#model)
    - [Evaluation](#evaluation)
- [Implementation](#implementation)
    - [Installation](#installation)
    - [Overview](#overview)
    - [Training](#training)
    - [Application](#app)
- [Future Versions](#future)

## About
## Data
### Source
- The training data set was sourced from [Inside Airbnb](https://insideairbnb.com/get-the-data/) and can be found on this [link](https://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2024-06-27/data/listings.csv.gz).

- Inside Airbnb provides quartely data for the last 12 months. It is possible, however, to make an archived data request, if necessary, on this [link](https://insideairbnb.com/data-requests/).

### Overview
#### Data Dictionary
- The features' name and detailed information can be found in the data dictionary provided by Inside Aribnb on this [link](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?gid=1322284596#gid=1322284596) 
- This project conforms to the definitions from August 2022.
#### Features
- Features relevant for the project:
    - accommodates
    - bathrooms
    - bedrooms
    - beds
    - minimum_nights_avg_ntm
    - availability_365
    - availability_90
    - number_of_reviews_ltm
    - review_scores_location
    - reviews_per_month
    - property_type
    - latitude
    - longitude
    - amenities
    - description
    - bathrooms_text
    - price

## Methodology
### Data Cleaning
### EDA and FE
### Feature Selection
### Model
### Evaluation
## Implementation
### Installation
*Python Version: 3.12.9*
1. Clone the repository
    ```sh
    git clone https://github.com/CarlosTussi/rio_listings.git
    ```
2. Change into the project directory
    ```sh
    cd rio-listings
    ```
3. (Recommended) Create a virtual environment
    ```sh
    python -m venv venv
    ```
4. (Recommended) Activate the virtual environment
    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```sh
      source venv/bin/activate
      ```
5. Install the required packages
    ```sh
    pip install -r requirements.txt
    ```
6. Download the training dataset [here](https://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2024-06-27/data/listings.csv.gz) if not already done.
7. Place the downloaded data inside the 'data' folder
    - Obs: rename it to 'listings.csv' if necessary.
### Usage (Windows)
- The training and the app run independently as the models are already pre-loaded in the repository.
- Should you wish to change the model, run the trining part first and then the app.
#### Training
1. Run the following command from the main folder 'rio_listings'
    ```sh
      python -m source.training.main
      ```
#### App
1. Run the following commands from the main folder 'rio_listings'
    ```sh
      $env:PYTHONPATH = (Get-Location)
      streamlit run .\source\app\main.py
      ```

### Overview
![alt text](https://github.com/CarlosTussi/rio_listings/blob/main/misc/diagram.png)
## Future Versions

