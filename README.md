
# SE ML Training Projects

Training pipeline of several projects, the main entry point of each project is ```main_PROJECT_NAME.py```.


## Author

- [@B-Yehuda](https://github.com/B-Yehuda)


## Environment Setup

All Python packages necessary to run the code in this repository are listed in `requirements.txt`. To create a new Anaconda environment which includes these packages, enter the following command in your terminal:

```bash
conda create --name ml_training --file requirements.txt
conda activate ml_training
```


## Code Execution
You will need Redshift credentials to run the code. Put the credentials in ```data/secrets/redshift_config.json```. **Make sure not to commit this file to the repository**.

To execute the training pipeline execute  the following command:
```
python main_PROJECT_NAME.py
```


## Running unit tests

To run the unit tests, simply run pytest in the tests directory.

```bash
cd tests
pytest
```


## Project Tree
```bash
├── main_acceptance_rate.py   
├── main_kronos.py   
├── main_youtube_conversions.py  

├── .github/workflows
  ├── docker-build-push.yml
  ├── unit-tests.yml

├── configs
  ├── config_clf_acceptance_rate.ini
  ├── config_clf_raid_tutorials.ini
  ├── config_reg_raid_tutorials.ini
  ├── config_clf_raid_deposits.ini
  ├── config_reg_raid_deposits.ini
  ├── config_clf_youtube_conversions.ini

├── src
  ├── __init__.py 
  ├── load_and_process_data.py
  ├── vault.py
  ├── evaluation_utilities
    ├── __init__.py 
    ├── loss_functions.py
    ├── model_evaluation.py
    ├── plot_utils.py
  ├── training_and_evaluation_pipeline  
    ├── __init__.py 
    ├── model_pipeline.py
    ├── model_training.py

├── tests
  ├── __init__.py
  ├── conftest.py
  ├── test_load_and_process_data.py
  ├── test_vault.py
  ├── test_evaluation_utilities
    ├── __init__.py 
    ├── test_loss_functions.py
    ├── test_model_evaluation.py
    ├── test_plot_utils.py
  ├── test_training_and_evaluation_pipeline 
    ├── __init__.py 
    ├── test_model_pipeline.py
    ├── test_model_training.py

├── .gitignore

├── Dockerfile

├── requirements.txt

├── README.md
```
