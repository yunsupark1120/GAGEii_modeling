# GAGEii Modeling

## 1. How to train the model

To train the model, follow the steps below:

1. Clone the repository

2. Have the NetCDF files stored in at the directory `data/time_series/{basin_id}.nc`

3. Have the static files stored in the directory `data/attributes/attributes.csv`

*This file should be present in the repository when cloning*

4. Configure the `config.yml` file with the desired parameters. For more information on the parameters, refer to the [Config.yml](#2-configyml) section. 
*Chnging the name of the file might result in errors*

5. Install the n`neuralhydrology` package by running the following command:
```bash
pip install neuralhydrology
```

6. Run the training script by running the following command:
```bash
python model_train.py
```

7. The model result will be stored in the directory `results/{run_name}`

## 2. Config.yml

Tune the following parameters in the `config.yml` file:

1. `model_name`: Name of the model

2. `{train/validation/test}_start_date`: start date for training, validation and testing in format `YYYY/MM/DD`

3. `{train/validation/test}_end_date`: end date for training, validation and testing in format `YYYY/MM/DD`

4. `{train/validation/test}_basin_file`: file directory for the list of the basin ids to be used for training, validation and testing

*This file should be formatted already when cloning the repository*

**The rest of the configuration settings are expected to be the same for all the models.**
