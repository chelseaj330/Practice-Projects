<h1> Company Emissions Pathways </h1>

This is a script that I wrote during my position as a researcher on the greenhouse gas emissions trajectories and the ambition of future emissions reduction for "non-state actors" (entities other than countries with substantial emissions, such as companies, cities, etc.).

There are two files, one for the main functions used in cleaning, selecting and transformation of the data and one containing the actual script.

The function file (Companies_functions.ipynb/.py) consists of three functions which:
1) Clean emissions data taken from two separate sources for each data year and calculates missing data
2) Use regression analysis to select which data source is best to use for each company in the dataset
3) Create time series for each company, using the selected data source, with 
    - interpolation between available emissions data points 
    - extrapolation, according to an imported growth rate, how a company's emissions would develop past their last recorded emissions reduction target

The script file (FinalScript_Companies.ipynb/.py) consists of the following steps:

- Setting parameters such as which reporting years' data to use and the time series growth rate to use as the basis for emissions interpolation
- Running each reporting year's data through the functions
- Aggregating the final emissions time series from 2020 - 2050 for each company into a single emissions time series per repoting year dataset
- Selecting companies that overlap (are present in all five year's of data) 
- Mapping each company to their relevant industrial sector as defined by the IEA
- Creating visualisations showing the resulting emissions time series for:
    1) All companies in each reporting year
    2) Overlapping companies in each reporting year
    3) Companies in each IEA sector and reporting year
- Creating summary tables that show information such as the number of companies, number of reported emissions reduction targets, number of targets within different time-scales and the average annual emissions reduction, for each reporting year
- Creating visualisations to check the distribution of certain variables and identify potential outliers


<h1> Installation & Usage </h1>

This script is only intended to showcase my abilities in Python scripting. It is not intended for external usage. For this reason, no data files are provided.

<h1> Contributing </h1>

Two files used in the main script, ProcessDuplicates.py and useful_functions.py, are not included in this directory, as these were written and contributed by a colleague working together on the project.

<h1> License </h1>

The script and code herein are intended for reading purposes only. I do not give my permission for them to be copied or otherwise used.
