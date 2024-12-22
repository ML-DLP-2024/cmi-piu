# CMI-PIU ML Project

- [CMI-PIU ML Project](#cmi-piu-ml-project)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Setup: Environment Variables](#setup-environment-variables)
    - [Setup: Kaggle API Key](#setup-kaggle-api-key)
  - [Setup: Install Libraries](#setup-install-libraries)
  - [Setup: Download the Data](#setup-download-the-data)
  - [Run](#run)

## Introduction

This is our Machine Learning (ML) project aimed at proposing a solution
to the Kaggle competition "Problematic Internet Use" held by the
Child Mind Institute.

<https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use>

We are a team of students at UET - VNU.

| #   | Student ID | Full Name         |
| --- | ---------- | ----------------- |
| 1   | 22025501   | Đỗ Trí Dũng       |
| 2   | 22028235   | Vũ Tùng Lâm       |
| 3   | 22028286   | Nguyễn Hữu Phương |

## Setup

### Setup: Environment Variables

In the project root, copy content of the`.env.example` file into
a new file named `.env`, then fill in the environment variables
as per the instructions.

### Setup: Kaggle API Key

The default data source (see below) is the original one, i.e. from
the competition itself on Kaggle. Your machine must have a Kaggle
API key file in order for this app to load data from this source.
Otherwise, you may have to define another data source. (More details
below.)

To get and install the Kaggle API key, refer to the instructions
at <https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials>.

## Setup: Install Libraries

Create a new Python virtual environment, activate
it and install the required packages.

```sh
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Setup: Download the Data

Run this command to list all the data sources.

```sh
python manage.py data sources
```

Example output:

```plain
+--------+-----------------------------------------------+
|  type  |                     name                      |
+--------+-----------------------------------------------+
| kaggle | child-mind-institute-problematic-internet-use |
+--------+-----------------------------------------------+
```

You can create another data source by writing a new class
inheriting BaseDataSource in `<project_root>/src/data/sources/DataSource`.
Refer to the `KaggleCompetitionDataSource.py` in that directory
for an example to get started.

Next, pull the data so that it is available for training.

```sh
python manage.py data pull "<source_type>" "<source_name>"
```

For example:

```sh
python manage.py data pull kaggle child-mind-institute-problematic-internet-use
```

Data will be downloaded to the path specified by the environment variable `DATA_DIR`.
You could check whether the data has been installed successfully by running:

```sh
python manage.py data history
```

To delete a data source:

```sh
python manage.py data delete "<source_type>" "<source_name>"
```

For example:

```sh
python manage.py data delete kaggle child-mind-institute-problematic-internet-use
```

## Run

List all the solutions:

```sh
python -m manage.py solutions list
```

Example output:

```plain
+-----------+
|   name    |
+-----------+
| solution1 |
+-----------+
```

Run the solution:

```sh
python -m manage.py solutions run solution1 --source-type kaggle --source-name child-mind-institute-problematic-internet-use
```
