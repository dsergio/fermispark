'''
Author: David Sergio
CSCD 530 Big Data Analytics Project
Winter 2025

# Project setup:
#
# Instructions: 
#
# 1 a) mkdir weekly_data
# 1 b) mkdir photon_events_files
# 1 c) mkdir output_data
# 1 d) mkdir binned_counts
# 2) cp set_environment_variables_template.sh set_environment_variables.sh and fill in the values
# 3) source set_environment_variables.sh
# 4) ./run_fermispark.sh


Steps:

1. Download the weekly photon and spacecraft data from the Fermi LAT data server, convert to CSV, and copy to HDFS
2. define spark mappers

For each subsequent step, we will use Spark with with a local Hadoop cluster

3. Read the photon data from HDFS
4. [spark] Compute the average energy per degree grid square (not used in subsequent steps, for demo purposes)
5. [spark] Compute the energy bins per degree grid square (not used in subsequent steps, for demo purposes)
6. [spark] Compute the energy counts per degree grid square. Set value in the energy bin to 1 if a photon is present in that bin. Save to CSV in HDFS and local file system. If there is already a CSV file with the same name (grid precision and bin width), then launch another map-reduce job to add the counts to the existing file, then save the updated file to HDFS and local file system.
7. Write the results to CSV files

'''

'''
Imports

'''
from pyspark import SparkConf, SparkContext
import os
from pyspark.sql import *
import pandas as pd
import numpy as np
import subprocess
import requests
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import csv
from astropy.io import fits
'''
Begin Configurations
'''

# Get environment variables
#
hdfs_master_ip = os.environ.get("FERMI_HDFS_MASTER_IP")
hdfs_master_port = os.environ.get("FERMI_HDFS_MASTER_PORT")
hdfs_dir = os.environ.get("FERMI_HDFS_DIR")
project_dir = os.environ.get("FERMI_PROJECT_DIR")
spark_master_ip = os.environ.get("FERMI_SPARK_MASTER_IP")
spark_master_port = os.environ.get("FERMI_SPARK_MASTER_PORT")

print("Environment variables:")
print(f"Project directory: {project_dir}")
print(f"HDFS directory: {hdfs_dir}")
print(f"HDFS master IP: {hdfs_master_ip}")
print(f"HDFS master port: {hdfs_master_port}")
print(f"Spark master IP: {spark_master_ip}")
print(f"Spark master port: {spark_master_port}")

# HDFS configurations
#

hdfs_server = f"hdfs://{hdfs_master_ip}:{hdfs_master_port}"

# data configurations
#
week = 865
grid_precision = 0
bin_width_MeV = 10000
max_energy_MeV = 400000 # 370989


overwrite_counts_file = False

num_partitions = 100

print_energy_bins = True
compute_average_energy = False

headers = ["RA", "DEC"]
for i in range(0, max_energy_MeV, bin_width_MeV):
    low = float(i) / 1000
    high = float(i + bin_width_MeV) / 1000
    headers.append(f"{low}-{high} GeV")

photon_events_file_name = "photon_events_data_" + str(week) + ".txt"

# project directories
#
weekly_data_dir = "weekly_data"
photon_events_dir = "photon_events_files"
binned_counts_dir = "binned_counts"
binned_counts_filename = f"binned_counts_precision_{grid_precision}_binwidth_{bin_width_MeV}_{max_energy_MeV}.csv"
output_dir = "output_data"
os.chdir(project_dir)

'''
End Configurations
'''

'''
Step 1. Download the weekly photon and spacecraft data from the Fermi LAT data server, convert to CSV, and copy to HDFS
 - a) Define download and copy functions. Download the photon data 
 - b) Convert the photon data to CSV
 - c) Copy the downloaded files to HDFS
'''

def get_weekly_file(week = 867):

    fits_file_name = "lat_photon_weekly_w" + str(week) + "_p305_v001"
    fits_file_name_spacecraft = "lat_spacecraft_weekly_w" + str(week) + "_p310_v001"

    fits_full_file_name = f"{project_dir}/{weekly_data_dir}/{fits_file_name}.fits"
    fits_full_file_name_spacecraft = f"{project_dir}/{weekly_data_dir}/{fits_file_name_spacecraft}.fits"

    ret = (fits_full_file_name, fits_full_file_name_spacecraft, fits_file_name, fits_file_name_spacecraft)

    print(fits_file_name)
    print(fits_file_name_spacecraft)

    weekly_photon_url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon/" + fits_file_name + ".fits"
    weekly_spacecraft_url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/" + fits_file_name_spacecraft + ".fits"

    if (os.path.exists(fits_full_file_name)):
        print("File exists")
        return ret
    else:
        print("Downloading file")
        r = requests.get(weekly_photon_url)
        open(fits_full_file_name, 'wb').write(r.content)
    
    if (os.path.exists(fits_full_file_name_spacecraft)):
        print("File exists")
        return ret
    else:
        print("Downloading file")
        r = requests.get(weekly_spacecraft_url)
        open(fits_full_file_name_spacecraft , 'wb').write(r.content)
    
    return ret


def fits_to_csv(fits_file, output_events_file):

    try:
        fits_f = fits.open(fits_file)
    except Exception as e:
        print(f"Error opening FITS file {fits_file}: {e}")
        return

    events = None
    for part in fits_f:
        extname = part.header.get('EXTNAME', '').upper()
        if extname == 'EVENTS':
            events = part

    events_data = events.data if events is not None else None

    columns = ["TIME", "RA", "DEC", "ENERGY"]

    try:
        with open(output_events_file, 'w', newline='') as csvfile:

            writer = csv.writer(csvfile)

            if events_data is not None:

                events_to_write = events_data
                writer.writerow(columns)

                for row in events_to_write:
                    row_data = [row[col] for col in columns]
                    writer.writerow(row_data)
            else:
                writer.writerow(["No EVENTS found."])

    except Exception as e:
        print(f"Error: {e}")

def copy_to_hdfs(local_path, hdfs_path):

    filename = os.path.basename(local_path)
    hdfs_full_path = f"{hdfs_path}/{filename}"
    
    # Check if file already exists in HDFS
    check_result = subprocess.run(
        ["hadoop", "fs", "-test", "-e", hdfs_full_path],
        capture_output=True,
        text=True
    )
    
    if check_result.returncode == 0:
        print(f"File {filename} already exists in {hdfs_path}, skipping copy")
        return

    print(f"HDFS Copying {local_path} to {hdfs_path}")
    result = subprocess.run(
        ["hadoop", "fs", "-put", "-f", local_path, hdfs_path],
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print(f"Successfully copied {local_path} to {hdfs_path}")
    else:
        print(f"Error: {result.stderr}")


fits_full_file_name, fits_full_file_name_spacecraft, fits_file_name, fits_file_name_spacecraft = get_weekly_file(week)
photon_events_local_data = f"{project_dir}/{photon_events_dir}/{photon_events_file_name}"

'''
1. b) Convert the photon data to CSV
'''
print("Converting photon data to CSV...")
fits_to_csv(fits_full_file_name, photon_events_local_data)



'''
1. c) Copy the downloaded files to HDFS
'''
print("Copying photon data to HDFS.")
photon_events_hdfs_data = f"{hdfs_server}{hdfs_dir}{photon_events_dir}/{photon_events_file_name}"
copy_to_hdfs(photon_events_local_data, f"{hdfs_server}{hdfs_dir}{photon_events_dir}")
print("photon events HDFS data: ", photon_events_hdfs_data)


'''
2. Define spark mappers
'''

def sky_grid_mapper(row, grid_precision = 0):
    ra = row["RA"]
    dec = row["DEC"]
    energy = row["ENERGY"]

    # 1x1 degree grid
    ra_int = int(ra)
    dec_int = int(dec)

    # higher precision grid
    if grid_precision > 0:
        ra_precision = round(ra, grid_precision)
        dec_precision = round(dec, grid_precision)
    else:
        ra_precision = ra_int
        dec_precision = dec_int

    return (ra_precision, dec_precision), (energy, 1)

def sky_grid_energy_bins_mapper(row, bin_width_MeV = 100000, grid_precision = 0):
    ra = row["RA"]
    dec = row["DEC"]
    energy = row["ENERGY"]

    # 1x1 degree grid
    ra_int = int(ra)
    dec_int = int(dec)

    # higher precision grid
    if grid_precision > 0:
        ra_precision = round(ra, grid_precision)
        dec_precision = round(dec, grid_precision)
    else:
        ra_precision = ra_int
        dec_precision = dec_int
        
    bucket_start = (energy // bin_width_MeV) * bin_width_MeV
    bucket_end = bucket_start + bin_width_MeV

    energy_bins_buckets = {(bucket_start, bucket_end): [str(energy)]}

    return (ra_precision, dec_precision), energy_bins_buckets

def sky_grid_energy_bins_count_mapper(row, bin_width_MeV = 100000, grid_precision = 0):
    ra = row["RA"]
    dec = row["DEC"]
    energy = row["ENERGY"]
    
    # 1x1 degree grid
    ra_int = int(ra)
    dec_int = int(dec)

    # higher precision grid
    if grid_precision > 0:
        ra_precision = round(ra, grid_precision)
        dec_precision = round(dec, grid_precision)
    else:
        ra_precision = ra_int
        dec_precision = dec_int
        
    bucket_start = (energy // bin_width_MeV) * bin_width_MeV
    bucket_end = bucket_start + bin_width_MeV

    energy_bins_buckets = {(bucket_start, bucket_end): ["1"]}

    return (ra_precision, dec_precision), energy_bins_buckets

def convert_list_to_csv_row(s, bin_width_MeV = 100000, max_energy_MeV = 400000):

    row = []
    for i in range(0, max_energy_MeV, bin_width_MeV):
        start = float(i)
        end = float(i + bin_width_MeV)
        key = (start, end)
        if (key in s[1]):
            
            list = s[1][key]
            row.append(';'.join(list))
        else:
            row.append("")
    
    ret = (s[0])

    for elem in row:
        ret += (str(elem),)

    return ret

def convert_list_counts_to_csv_row(s, bin_width_MeV = 100000, max_energy_MeV = 400000):

    row = []
    for i in range(0, max_energy_MeV, bin_width_MeV):
        start = float(i)
        end = float(i + bin_width_MeV)
        key = (start, end)
        if (key in s[1]):
            list = s[1][key]
            row.append(';'.join(list))
        else:
            row.append("")
        
    
    ret = (s[0])
    
    for elem in row:
        sum = 0
        for i in elem.split(';'):
            if (i != ""):
                sum += int(i)
        ret += (sum,)
    return ret


with SparkSession.builder \
    .appName("Fermispark") \
    .config("spark.master", f"spark://{spark_master_ip}:{spark_master_port}") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate() as spark:

    sc = spark.sparkContext
    sc.setLogLevel("INFO")
    sc.setLogLevel("WARN")
    # sc.setLogLevel("ERROR")


    '''
    3. Read the photon data from HDFS
    '''

    photon_counts = spark.read.csv(photon_events_hdfs_data, header=True, inferSchema=True)
    photon_counts_partitions = photon_counts.repartition(num_partitions)
    photon_counts_partitions.cache()

    print("Photon counts partitions: ", photon_counts_partitions.rdd.getNumPartitions())
    print("photon counts schema: \n")
    photon_counts_partitions.printSchema()
    photon_counts_partitions.show(10)


    '''
    4. [spark] Compute the average energy per degree grid square
    Just for demo purposes. Optional
    '''

    if compute_average_energy:

        average_energy_per_grid_sorted = photon_counts_partitions.rdd \
            .map(lambda row: sky_grid_mapper(row, grid_precision)) \
            .reduceByKey(lambda a, b: (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]))) \
            .mapValues(lambda s: float(s[0]) / float(s[1])) \
            .sortBy(lambda s: s[0][0], ascending=True) \
            .sortBy(lambda s: s[0][1], ascending=True) \
            .collect()

        average_energy_bins = pd.DataFrame(average_energy_per_grid_sorted, columns=["coord", "average_energy"])
        print(average_energy_bins)

    '''
    5. [spark] Compute the energy bins per degree grid square
    Save to CSV. Optional
    '''

    if print_energy_bins:

        binnned_energy_per_grid = photon_counts_partitions.rdd \
            .map(lambda row: sky_grid_energy_bins_mapper(row, bin_width_MeV, max_energy_MeV)) \
            .reduceByKey(lambda a, b: {k: a.get(k, []) + b.get(k, []) for k in set(a) | set(b)}) \
            # .sortBy(lambda s: s[0][1], ascending=True) \
            # .collect()

        # energy_bins = pd.DataFrame(binnned_energy_per_grid, columns=["coord", "energy_bins"])
        # print(energy_bins)


        binnned_energy_per_grid_csv = binnned_energy_per_grid.map(lambda s: convert_list_to_csv_row(s, bin_width_MeV, max_energy_MeV)) \
            .sortBy(lambda s: (s[1], s[0]), ascending=True) \
            .collect() 

        binnned_energy_per_grid_csv = pd.DataFrame(binnned_energy_per_grid_csv, columns=headers)
        binned_energy_per_grid_csv_filename = f"{project_dir}/{output_dir}/output_binned_week_{week}_precision_{grid_precision}_binwidth_{bin_width_MeV}_{max_energy_MeV}.csv"
        print(f"Printing binned photon energies for week {week} to file {binned_energy_per_grid_csv_filename}\n", binnned_energy_per_grid_csv)
        binnned_energy_per_grid_csv.to_csv(binned_energy_per_grid_csv_filename, index=False)

    # quit() # Testing the above code

    '''
    6. [spark] Compute the energy counts per degree grid square
    Save to CSV
    '''

    binnned_energy_counts_per_grid = photon_counts_partitions.rdd \
        .map(lambda row: sky_grid_energy_bins_count_mapper(row, bin_width_MeV, max_energy_MeV)) \
        .reduceByKey(lambda a, b: {k: a.get(k, []) + b.get(k, []) for k in set(a) | set(b)}) \
        # .sortBy(lambda s: s[0][1], ascending=True) \
        # .collect()

    # Unpersist cached RDDs, sometimes causes network errors
    # 
    photon_counts_partitions.unpersist()

    # counts_df = pd.DataFrame(binnned_energy_counts_per_grid, columns=["coord", "energy_bins"])
    # print(counts_df)

    hdfs_binned_counts_output_path = f"{hdfs_server}{hdfs_dir}{binned_counts_dir}/{binned_counts_filename}"
    counts_csv_week_df_filename = f"{project_dir}/{output_dir}/output_binned_counts_week_{week}_precision_{grid_precision}_binwidth_{bin_width_MeV}_{max_energy_MeV}.csv"
    counts_csv_cumulative_df_filename = f"{project_dir}/{output_dir}/output_binned_counts_precision_{grid_precision}_binwidth_{bin_width_MeV}_{max_energy_MeV}.csv"

    binnned_energy_counts_per_grid_csv_week = binnned_energy_counts_per_grid \
            .map(lambda s: convert_list_counts_to_csv_row(s, bin_width_MeV, max_energy_MeV)) \
            .map(lambda x: tuple(round(float(i), grid_precision) for i in x)) \
            .sortBy(lambda s: (s[1], s[0]), ascending=True)
    
    
    schema = StructType([StructField(header, FloatType(), True) for header in headers])

    no_existing_binned_counts = False

    if overwrite_counts_file:
        print(f"Overwriting counts for week {week} to file {counts_csv_cumulative_df_filename}")
    else:
        print(f"Adding counts for week {week} to file {counts_csv_cumulative_df_filename}")

        try:
            existing_binned_counts = spark.read.csv(hdfs_binned_counts_output_path, header=False, schema=schema)
        except:
            print("No existing binned counts file found. Let's create a new one.")
            existing_binned_counts = spark.createDataFrame([], schema)

        if (existing_binned_counts.rdd.isEmpty()):
            no_existing_binned_counts = True

        else:
            pass

        existing_binned_counts_partitions = existing_binned_counts.repartition(num_partitions)

        def sum_lists(a, b):
            return [a[i] + b[i] for i in range(len(a))]
        
        df = spark.createDataFrame(binnned_energy_counts_per_grid_csv_week, schema)

        binnned_energy_counts_per_grid_csv_cum = df.union(existing_binned_counts_partitions)
    
        binnned_energy_counts_per_grid_csv_cum = binnned_energy_counts_per_grid_csv_cum.rdd \
            .map(lambda x: ((x[0], x[1]), list(x[2:]))) \
            .reduceByKey(sum_lists) \
            .map(lambda x: (x[0][0], x[0][1], *x[1])) \
            .sortBy(lambda s: (s[1], s[0]), ascending=True)
    

    hdfs_df = spark.createDataFrame(binnned_energy_counts_per_grid_csv_cum, schema)
    hdfs_df.write.mode("overwrite").csv(hdfs_binned_counts_output_path)

    binnned_energy_counts_per_grid_csv_cum = binnned_energy_counts_per_grid_csv_cum.collect() 

    binnned_energy_counts_per_grid_csv_week = binnned_energy_counts_per_grid_csv_week.collect() 

    counts_csv_df_week = pd.DataFrame(binnned_energy_counts_per_grid_csv_week, columns = headers)
    counts_csv_df = pd.DataFrame(binnned_energy_counts_per_grid_csv_cum, columns = headers)

    print(f"Writing weekly counts for week {week} to file {counts_csv_week_df_filename}\n", counts_csv_df_week)
    counts_csv_df_week.to_csv(counts_csv_week_df_filename, index=False)

    print(f"Writing cumulative counts including week {week} to file {counts_csv_cumulative_df_filename}\n", counts_csv_df)
    counts_csv_df.to_csv(counts_csv_cumulative_df_filename, index=False)

