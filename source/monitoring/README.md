# Monitoring

- This monitoring module consists of comparing the model performance of two different datasets with the models that were previously trained.

- The two datasets are:
    * **Reference Data**: any validated older version of the dataset.
    * **Current Data**: New dataset (ground truth) to check how much the model has degraded since last update.

- To generate the report, module was integrated with [Evidently AI](https://www.evidentlyai.com/).

>_Important: This initial version will fetch a new data set from the web with the provided URL given by the user. For the Reference Dataset, it uses by default the training dataset in the [data folder](https://github.com/CarlosTussi/rio_listings/tree/main/data)_.

## How to use it
1) Make sure that the models are trained and exist in the [models folder](https://github.com/CarlosTussi/rio_listings/tree/main/models).

2) Make sure you have the training (reference data) in the [data folder](https://github.com/CarlosTussi/rio_listings/tree/main/data).

3) From the rio_listings foldder run:
    ```sh
    python -m  source.monitoring.main [NEW_DATA_SET_URL]
    ```

    * _OBS 1: For this version, the dataset needs to be specifically compressed in a .gz file from Inside Airbnb. This is an example of conforming URL in the correct format_ https://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2024-12-27/data/listings.csv.gz

    * _OBS 2: The features in the new dataset needs to conform with the project specification._

4) Web browser will automatically open displaying the HTML report generating comparing the model performance with the two datasets.

    *_OBS: Should the web browser fails to display the HTML page, a copy of the HTML file is generated in the [monitoring folder](https://github.com/CarlosTussi/rio_listings/tree/main/source/monitoring) at the end of the execution._