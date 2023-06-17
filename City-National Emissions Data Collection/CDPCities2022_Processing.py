#!/usr/bin/env python
# coding: utf-8

# <h1> Comparison of City-National Climate Change Mitigation Policies: Data Processing & Compiling </h1>

# This is a script I've used on a project where I needed to gather data on the greenhouse gas emissions, climate change mitigation actions, emissions reduction targets and mitigation policies for cities and their respective countries. I gathered, cleaned and aggregated the data for four city-country pairs as case studies. These were then exported to separate Excel files as 'factsheets' for further analysis.
# 
# City data on the following aspects from <b><i> CDP's 2022 Cities Questionnaire Response Data </i></b> (publicly available [via their data portal](https://data.cdp.net/browse))
# - _City emissions inventories_
# - _City emissions reduction actions/measures_
# - _Drafted city mitigation plans_
# - _City emissions reduction targets_
# - _City population (reported within the previously listed datasets)_
# 
# National data on the following aspects and from the following sources:
# - _National emissions inventories_ (CAIT data taken from [Climate Watch platform](https://www.climatewatchdata.org/ghg-emissions?end_year=2019&start_year=1990); separate files downloaded for each country needed)
# - _National emissions reduction targets_ (manually compiled into a csv file based on info from the [Climate Action Tracker](https://climateactiontracker.org/))
# - _National climate change policies_ (dataset downloaded from New Climate Institute's [Climate Policy Database](https://climatepolicydatabase.org/))
# - _National population_ (dataset downloaded from the [World Bank](https://data.worldbank.org/indicator/SP.POP.TOTL))

# In[1]:


import numpy as np
import pandas as pd

## City data files to be imported; downloaded from CDP's 2022 Cities Questionnaire Data as separated datasets:
city_files = ['20220424_Cities_Measures','20220424_Cities_Targets', '20220424_Cities_Emissions', '20220424_Cities_Mitigation_Plans']

dir_data = 'City_Country_Data/'

city_frames = []

for i in range(len(city_files)):
    ## Read each file of city data
    c_df = pd.read_csv(r''+dir_data+city_files[i]+'.csv', on_bad_lines='skip', sep=',')
    ## Drop these unnecessary columns from each dataset
    c_df.drop(['CDP Region', 'Access', 'C40 City', 'GCoM City'], axis=1, inplace=True)
    ## Append to empty list for further processing
    city_frames.append(c_df)
    
city_mitactions = city_frames[0]
city_targets = city_frames[1]
city_emissions = city_frames[2]
city_mitplans = city_frames[3]


# <h1> City Data: Cleaning </h1>

# <h2> Emissions Data </h2>

# In[2]:


## Check data available in the dataset
city_emissions.columns


# In[3]:


## Dropping remaining unnecessary columns
city_emissions.drop(['Emissions Question Number','Emissions Question Name','Emissions Column Number', 'Emissions Notation Key', 'Emissions Data Reported', 'Emissions Rank'], axis=1, inplace=True)

## Sort by emissions category
city_emissions.sort_values(by=['Emissions Row Number'],  inplace=True)

## Filter our this emissions category
city_emissions = city_emissions[city_emissions['Emissions Column Name'] != 'Emissions occurring outside the jurisdiction boundary as a result of in-jurisdiction activities (metric tonnes CO2e)']

## Filter for certain types of emissions data groups: sector and source totals for each category, as well as any category related to energy or buildings
city_emissions = city_emissions[((city_emissions['Emissions Data Group'] == 'SectorTotal') | (city_emissions['Emissions Data Group'] == 'SourceTotal') |
(city_emissions['Emissions Data Group'].str.contains('grid', case=False))) | (city_emissions['Emissions Description'].str.contains('building', case=False))]


# In[4]:


## Need to clean: Emissions were entered in the data set as a text field with commas as thousands separator
## Remove commas
city_emissions['Emissions Response Answer'] = city_emissions['Emissions Response Answer'].str.replace(',','', regex=False)
## Convert to numerical data type
city_emissions['Emissions Response Answer'].astype('float64', copy=False)


# <h2> Emissions Reduction Actions </h2>

# In[5]:


## Check available data
city_mitactions.columns


# In[6]:


## Check available sector for emissions reduction actions
city_mitactions['Primary emissions sector'].value_counts()


# We can see here the economic sectors into which emissions reduction actions are categorized. For my work, I was interested in measures related to energy, waste, transportation and buildings. However, we can see here that buildings is not disaggregated as a separate category. Generally in emissions inventories, buidlings is a subsector of energy. Below I check the subcategories for rows where Stationary Energy is listed as the primary emissions sector.

# In[7]:


## Check the most common action types for energy sector emissions reduction actions
energy_action_types = pd.DataFrame(city_mitactions[city_mitactions['Primary emissions sector'] == 'Stationary energy']['Action type'].value_counts())

energy_action_types.head(20)


# We can see there are several subcategories of measures in the Stationary Energy sector that relate to buildings, including the most common subcategory relating to energy efficiency in buildings. I will therefore filter these out and re-assign them as a separate category in the steps below.

# In[8]:


sectors = ['Stationary energy', 'Generation of grid-supplied energy', 'Transportation', 'Waste']

## Filter emissions reduction actions data for actions:
## - In the above sectors
## - or, even if not in the above sectors, actions related to buildings
## - only actions that are in the implementation or operation phase
mitactions_filtered = city_mitactions[((city_mitactions['Primary emissions sector'].isin(sectors)) | (city_mitactions['Action type'].str.contains('building', case=False)) | (city_mitactions['Action type'].str.contains('buildings', case=False)) |
                                    (city_mitactions['Action description and web link'].str.contains('building', case=False)) | (city_mitactions['Action description and web link'].str.contains('buildings', case=False))) &
                                     ((city_mitactions['Status of action in the reporting year'].str.contains('implementation', case=False)) | 
                                   (city_mitactions['Status of action in the reporting year'].str.contains('operation', case=False)))]

## Drop remaining unnecessary columns
mitactions_filtered.drop(['Estimated annual energy savings (MWh/year)', 'Estimated annual renewable energy generation (MWh/year)'], axis=1, inplace=True)                      


# In[9]:


## Buildings is listed as a subsector within Stationary Energy
## Create new primary sector definition for Buildings from the rows where Action type contains Buildings; keep other primary sector definitions
mitactions_filtered.reset_index(inplace= True)

for index, row in mitactions_filtered.iterrows():
    # convert the 'Action type' column to a string type
    action_type_str = str(row['Action type'])
    if 'building' in action_type_str.lower():
        mitactions_filtered.at[index, 'URBAN SECTOR'] = 'Buildings'
    else:
        mitactions_filtered.at[index, 'URBAN SECTOR'] = mitactions_filtered.at[index, 'Primary emissions sector']

## Check result
mitactions_filtered


# <h3> Pulling 2021 Data for Means of Implementation </h3>

# CDP changed their questionnaire between 2022 and 2021. In the 2021 questinnaire, they asked cities to categorize their emissions reduction action from a drop down menu of 'means of implementation'. However, this information is no longer available in the 2022 response data. So I import here the 2021 Emissions Reduction Action dataset and match the information for means of implementation from those measures if they were also reported in 2022.

# In[10]:


## 2021 Emissions Reduction Action data from CDP is used to fulfill the following data needs:
## 1) Categorization of some actions by their "means of implementation" (if they were reported by a city in both 2021 and 2022)
## 2) Action data for Den Haag, because not enough useful informationw was provided in reporting on their actions from 2022

mitactions_2021 = pd.read_csv(r''+dir_data+'2021_Cities_Emissions_Reduction_Actions.csv')


# In[11]:


citynames_2021 = ['Balikpapan City Government', 'City of Lakewood, CO', 'City of Buenos Aires', 'Gemeente Den Haag']

sectoral_mitactions = mitactions_2021[((mitactions_2021['Mitigation Action'].str.contains('energy', case=False)) | (mitactions_2021['Mitigation Action'].str.contains('buildings', case=False)) |
                         (mitactions_2021['Mitigation Action'].str.contains('waste', case=False)) | (mitactions_2021['Mitigation Action'].str.contains('transport', case=False))) &
                                       ((mitactions_2021['Implementation Status'] == 'Implementation') | (mitactions_2021['Implementation Status'] == 'Operation'))]

# case_mitactions = sectoral_mitactions[(sectoral_mitactions['Organization'].isin(citynames_2021))]
                                       
nodupes_mitactions = sectoral_mitactions[sectoral_mitactions['Action Title'].duplicated(keep='first') == False]

nodupes_mitactions.drop(columns='Access', inplace=True)


# In[12]:


temp_final = nodupes_mitactions[nodupes_mitactions['Organization'].isin(citynames_2021)]
name_dupes = sectoral_mitactions[(sectoral_mitactions['Organization'].isin(citynames_2021)) & (sectoral_mitactions['Action Title'].duplicated(keep='first') == True)]
name_dupes_grouped = name_dupes.groupby(['Action Title'], as_index=False).agg({'Means of Implementation': ', '.join})
## Found solution to group by with string columns here: https://www.statology.org/pandas-groupby-concatenate-strings/
name_dupes_grouped.rename(columns={'Means of Implementation':'Duplicate Means of Implementation'}, inplace=True)
name_merged = pd.merge(temp_final, name_dupes_grouped, how='left', on=['Action Title'])
## Explanation of join types: https://pandas.pydata.org/docs/user_guide/merging.html
name_merged['Means of Implementation: All'] = name_merged['Means of Implementation'] + ', ' + name_merged['Duplicate Means of Implementation']
name_merged.drop(columns=['Means of Implementation', 'Duplicate Means of Implementation'], inplace=True)

## Check result
name_merged[['Action Title', 'Means of Implementation: All']]


# <h2> Mitigation Plans </h2>

# In[13]:


## Check available data
city_mitplans.columns

## No further processing needed


# <h2> Emissions Reduction Targets </h2>

# In[15]:


## Check available data
city_targets.columns

## No futher processing needed


# <h1> National Data: Cleaning & Aggregation </h1>

# In[16]:


# country_isos = ['IDN', 'USA', 'ARG', 'NLD', 'EU']

# countries = ['Indonesia','United States', 'Argentina', 'Netherlands', 'EU']

## Import datasets of national emissions for each selected country
cait_files = ['IDN_CAIT','USA_CAIT','ARG_CAIT', 'NLD_CAIT']

cait_frames = []

for i in range(len(cait_files)):
    cait_df = pd.read_csv(r''+dir_data+cait_files[i]+'.csv', on_bad_lines='skip', sep=',')
    ## Only take most relevant columns: emissions sector, unit of measure, most recent emissions years
    cait_new = cait_df.iloc[:-2,:].loc[:,['Sector','unit','2018','2019']]
    ## Add country code column for easier recognition when datasets are concatenated
    cait_new['Country Code'] = cait_files[i]
    cait_frames.append(cait_new)

## Concatenate emissions data for all selected countries to one dataset
cait_full = pd.concat(cait_frames)

## Import dataset of climate change related national policies downloaded from New Climate Institute's Climate Policy Database
nc_preselected = pd.read_csv(r''+dir_data+'ALL_NC_Policies.csv')
## Filter for only policies that are in force
nc_inforce = nc_preselected[nc_preselected['Implementation state'] == 'In force']

## Import World Bank population data
wb_pop = pd.read_excel(r''+dir_data+'WorldBank_Population.xlsx', sheet_name='Data')

## Import dataset of national emissions reduction targets manually gathered for selected countries
natl_targets =  pd.read_excel(r''+dir_data+'National_Targets.xlsx')


# <h1> Combined Data Aggregation & Export </h1>

# In[18]:


## Set up list with tuples containing each city's official name, country's ISO code and country name
citycountry_pairs = [('City of Buenos Aires', 'ARG', 'Argentina'), ('City of Lakewood, CO', 'USA', 'United States'), ('Balikpapan City Government', 'IDN', 'Indonesia'), 
                     ('Municipality of The Hague', 'NLD', 'Netherlands')]

## Loop over city-country pair list, taking information from each dataset that matches each city, iso and country name, respectively
for city, iso, country in citycountry_pairs: 
        with pd.ExcelWriter(r'TEST_'+city+'_Factsheet.xlsx') as writer:
            city_emissions[city_emissions['Organization Name'] == city].to_excel(writer, sheet_name='City Emissions')
            city_targets[city_targets['Organization Name'] == city].to_excel(writer, sheet_name='City Targets')
            mitactions_filtered[mitactions_filtered['Organization Name'] == city].to_excel(writer, sheet_name='City Mitigation Actions')
            city_mitplans[city_mitplans['Organization Name'] == city].to_excel(writer, sheet_name='City Mitigation Plan')
            cait_full[cait_full['Country Code'].str.contains(iso)].to_excel(writer, sheet_name='National Sectoral Emissions')
            nc_inforce[nc_inforce['Country ISO'] == iso].to_excel(writer, sheet_name='National Sectoral Policies')
            wb_pop[wb_pop['Country Code'] == iso].to_excel(writer, sheet_name='National Population')  
            natl_targets[natl_targets['Country'] == iso].to_excel(writer, sheet_name='National Targets')

