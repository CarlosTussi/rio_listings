Airbnb Propeprty Price Predictor 

Data sourced from [Inside Airbnb](https://insideairbnb.com/get-the-data/)

## Table of Contents
- [About](#about)
- [Data](#data)
    * [Source](#source)
    * [Overview](#overview)
        + [Features](#features)
- [Methodology](#methodology)
    * [Data Cleaning](#datacleaning)
    * [EDA/FE](#edafe)
    * [Feature Selection](#featureselection)
    * [Model](#model)
    * [Evaluation](#evaluation)
- [Implementation](#implementation)
    * [Installation](#installation)
    * [Training](#training)
    * [Application](#app)
- [Future Versions](#future)

## Installation
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

## Usage

Run the script indicating the number of pages to be scraped.

```sh
python .\scraper.py 2
```

## Output

CSV file ("books.csv") is generated with the following headers:
*  Title	- Price  - 	Stars

Each row will contain the relevant data for a book extracted from the website.

