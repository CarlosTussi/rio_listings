# Containers

In case there are difficulties or compatibility issues in running this project, a container was created to address these potential problems, allowing the training module to generate the models.

## How to run the container
1. Clone this repository
    ```sh
    git clone https://github.com/CarlosTussi/rio_listings.git
    ```
2. Install Docker if not yet done.
    * Instructions [here](https://www.docker.com/get-started/).
3. Start Docker deamon
    * One easy option is to simply open the Docker Desktop app downloaded before.
4. Pull the container
    ```sh
    docker pull cetleite/training-image:latest
    ```
5. Make sure the data was downloaded as instructed [here](https://github.com/CarlosTussi/rio_listings/tree/main/data).
6. From inside [rio_listings](https://github.com/CarlosTussi/rio_listings/tree/main) main folder run:
    ```sh
    docker run -v "$(pwd)/data:/training_root/data" -v "$(pwd)/containers/models:/training_root/models" cetleite/training-image
    ```
    Obs: _There are two volume bind-mount: one to make the data available for the container and another to be able to access the trained model (output of the training module)._

7. The trained models will be inside models folder inside [containers folder](https://github.com/CarlosTussi/rio_listings/tree/main/containers).

## Creating as new image
Should you wish to alterate the code and create a new image, follow the following steps.

1. Clone this repository
    ```sh
    git clone https://github.com/CarlosTussi/rio_listings.git
    ```
2. Install Docker if not yet done.
    * Instructions [here](https://www.docker.com/get-started/).
3. Start Docker deamon
    * One easy option is to simply open the Docker Desktop app downloaded before.
4. Change the [Dockerfile](https://github.com/CarlosTussi/rio_listings/tree/main/containers/Dockerfile) located inside **containers** folder if necessary.
    * If no new files/folders were created and if no paths, dependencies or python version were changed, then you can probably use the same Dockerfile. 

5. From inside **containers** folder where the [Dockerfile](https://github.com/CarlosTussi/rio_listings/tree/main/containers/Dockerfile) is located, run:
    ```sh
    docker build -f Dockerfile -t [YOUR-IMAGE-NAME] ..
    ```

    * _Obs: Replace [YOUR-IMAGE-NAME] for a suitable name._

6. From inside [rio_listings](https://github.com/CarlosTussi/rio_listings/tree/main) main folder run:
    ```sh
    docker run -v "$(pwd)/data:/training_root/data" -v "$(pwd)/containers/models:/training_root/models" [YOUR-IMAGE-NAME]
    ```

    * _Important: Adapt the docker run command accordingly. If you need to add/remove any binding do so as needed._