# Parallel Demand Forecasting at Scale using Ray Tune and Ray Data

This example demonstrates how to perform large-scale parallel model training and hyperparameter tuning for time series forecasting using Ray and NeuralProphet on Databricks. The solution leverages Ray's distributed computing capabilities to efficiently train many individual forecasting models in parallel.

## Overview

Batch training and tuning are common machine learning tasks, particularly in forecasting scenarios where separate models are required for different products, locations, or other entities. While training these models sequentially is simple, it can become prohibitively time-consuming as the number of models increases. This example showcases how to efficiently parallelize this process using Ray.

<div align="center">
  <img src="https://docs.ray.io/en/master/_images/batch-training.svg" alt='Batch Training Architecture' height="300" width="400">
</div>

## Dataset

This example uses the [Walmart M5 Forecasting Dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy), which contains historical sales data for products across different Walmart stores. The dataset provides hierarchical information about:

- Products (across departments and categories)
- Store locations (across states)
- Daily sales over a period of time

The example demonstrates how to train many forecasting models (one per product/store combination) in parallel, significantly reducing the overall time required for training.

## Prerequisites

- Databricks ML Runtime 15.4 or higher


## Key Components

This example consists of two notebooks:

1. **01_preparing_walmart_m5_data.ipynb**: Prepares the Walmart M5 dataset for time series forecasting by:
   - Downloading and unzipping the data from Kaggle
   - Processing the raw files into a suitable format for forecasting
   - Storing the processed data in Delta tables

2. **main.ipynb**: Implements the parallel forecasting workflow:
   - Sets up Ray on Databricks
   - Partitions the data for individual model training
   - Defines the NeuralProphet model training function
   - Uses Ray Tune for hyperparameter optimization
   - Tracks model performance with MLflow
   - Visualizes results and forecasts

## Getting Started

1. Run the `01_preparing_walmart_m5_data.ipynb` notebook to download and prepare the Walmart M5 dataset
2. Run the `main.ipynb` notebook to execute the parallel forecasting workflow

Note: You will need to have a Kaggle account and API credentials to download the dataset.

## Architecture

This solution demonstrates the following key capabilities:

- **Ray Data**: For efficient data partitioning and distribution across workers
- **Ray Tune**: For distributed hyperparameter optimization
- **MLflow**: For experiment tracking and model management
- **NeuralProphet**: For time series forecasting using neural networks

## Benefits

- **Scalability**: Train hundreds or thousands of forecasting models in parallel
- **Performance**: Significantly reduce end-to-end training time
- **Resource Efficiency**: Efficiently utilize available compute resources
- **Flexibility**: Easily adapt the approach to other time series forecasting problems

## Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [NeuralProphet Documentation](https://neuralprophet.com/)
- [Databricks-Ray Integration Guide](https://docs.databricks.com/aws/en/machine-learning/ray/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Video Demonstration

Watch this video from the Data and AI Summit 2023, where Marks & Spencer's talk about how they scaled their demand forecasting use case using the above approach:

[![Parallel Demand Forecasting with Ray and Databricks](https://img.youtube.com/vi/H5ToDhX4Uqg/0.jpg)](https://www.youtube.com/watch?v=H5ToDhX4Uqg)

This video demonstrates:
- Setting up the environment
- Data preparation and processing
- Parallel model training with Ray
- Hyperparameter tuning
- Results visualization

