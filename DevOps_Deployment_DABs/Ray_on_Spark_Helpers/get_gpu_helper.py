# Databricks notebook source
# MAGIC %pip install bs4 lxml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

def get_all_gpu_data(providers=['aws', 'gcp', 'azure']):
    """
    Scrapes GPU instance data for specified cloud providers and combines it
    into a single pandas DataFrame.

    Args:
        providers (list, optional): A list of providers to scrape.
                                    Defaults to ['aws', 'gcp', 'azure'].

    Returns:
        A pandas DataFrame containing all instance types, or an empty DataFrame if failed.
    """
    # A single list will hold all DataFrames before the final concatenation.
    list_of_dataframes = []

    for provider in providers:
        try:
            # Step 1: Determine the correct URL based on the provider.
            if provider == 'azure':
                url = "https://learn.microsoft.com/en-us/azure/databricks/compute/gpu"
            elif provider in ['aws', 'gcp']:
                url = f"https://docs.databricks.com/{provider}/en/compute/gpu.html"
            else:
                print(f"Warning: Provider '{provider}' is not recognized. Skipping.")
                continue

            print(f"Requesting page data from: {url}")
            response = requests.get(url)
            response.raise_for_status()
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # Step 2: Apply the correct scraping strategy for the provider.
            # --- STRATEGY FOR AWS & GCP (Tab-based layout) ---
            if provider in ['aws', 'gcp']:
                tabs = soup.select("li[role='tab']")
                panels = soup.select("div[role='tabpanel']")

                if not tabs or len(tabs) != len(panels):
                    print(f"Warning: Page structure mismatch for {provider.upper()}. Skipping.")
                    continue

                for i, panel in enumerate(panels):
                    table_tag = panel.find('table')
                    if table_tag:
                        df = pd.read_html(str(table_tag))[0]
                        df['Provider'] = provider.upper()
                        # df['Series'] = tabs[i].get_text(strip=True)
                        list_of_dataframes.append(df)

            # --- STRATEGY FOR AZURE (Heading-based layout) ---
            elif provider == 'azure':
                headings = soup.find_all('h4')
                for heading in headings:
                    table_tag = heading.find_next('table')
                    if table_tag:
                        df = pd.read_html(str(table_tag))[0]
                        df['Provider'] = provider.upper()
                        # df['Series'] = heading.get_text(strip=True)
                        list_of_dataframes.append(df)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the request for {provider.upper()}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {provider.upper()}: {e}")

    # Step 3: Combine all collected data.
    if not list_of_dataframes:
        print("No data was collected.")
        return pd.DataFrame()

    combined_df = pd.concat(list_of_dataframes, ignore_index=True)
    print("\nAll tables combined successfully! ✅")
    return combined_df


# Call the single function to get data for all providers.
all_providers_df = get_all_gpu_data()

if not all_providers_df.empty:
    # Reorder columns for better readability.
    cols_to_move = ['Provider']
    all_providers_df = all_providers_df[[col for col in cols_to_move if col in all_providers_df.columns] + [col for col in all_providers_df.columns if col not in cols_to_move]]
    
    print("\nCombined DataFrame for All Providers:")
    print(all_providers_df.to_string())