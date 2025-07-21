
# 🌿 Stochastic Ecohydrological Rainfall–Runoff Modeling

This repository contains code for for a semi-distributed stochastic ecohydrological model for simulating watershed processes. It integrates storm event rainfall-runoff, vegetation-driven evapotranspiration, and soil moisture dynamics to simulate runoff generation, baseflow, and evapotranpsiration at the watershed scale. The modeling framework is designed for scenario analysis, calibration with USGS data, and integration with remote sensing and reanalysis datasets.

The model and supporting code were developed by Mark S. Bartlett and Elizabeth Cultra and Amilcare Porporato as part of ongoing research in probabilistic ecohydrology and watershed-scale hydrologic prediction.

---

## 📁 Repository Structure

The respository contains the code for aquiring the data (Baseflow.py, Daymet Data.py and ET_MODIS_data.py), processing the data for the model (Model_Continuous Parameter_Data - FL.py, Model_Continuous Parameter_Data - LA.py,  Model_Long_Term_Calibration_Parameters_FL.py, and  Model_Long_Term_Calibration_Parameters_LA.py # Long-term calibration - LA), and calibrating the model (Model_Calibration.nb ). The data outputs (for the processed data) are in `reports/data`, and the figures of the recent paper 'Stochastic ecohydrological perspective on 
semi-distributed rainfall-runoff dynamics' may be recreated with the files in `reports/figures`. The code is setup to work with AWS s3 storage, and will need to be modified accordingly to redirect the data storage to a different location.

```
stoch-ecohydro-runoff/
└── stoch-ecohydro-runoff-main/
    ├── notebooks/                  # Scripts and notebooks for data setup, model calibration, and ET data preparation
    │   ├── Baseflow_Data.py                        # Baseflow extraction and processing
    │   ├── CN Values Traditional.ipynb             # Curve Number analysis
    │   ├── DayMet Data.py                          # Processing of DayMet rainfall input
    │   ├── ET_MODIS_Data.py                        # MODIS ET data integration
    │   ├── Model_Calibration.nb                    # Mathematica notebook for parameter fitting
    │   ├── Model_Continuous Parameter_Data - FL.py # Calibration for Florida site
    │   ├── Model_Continuous Parameter_Data - LA.py # Calibration for Louisiana site
    │   ├── Model_Long_Term_Calibration_Parameters_FL.py # Long-term calibration - FL
    │   └── Model_Long_Term_Calibration_Parameters_LA.py # Long-term calibration - LA
    ├── reports/                    # Data and figures used in analysis and publication
    │   ├── data/
    │   │   ├── USGS_gage_event_rainfall_jacksonville.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   │   ├── USGS_gage_event_rainfall_lwi-transition-zone.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   │   ├── USGS_gage_event_runoff_jacksonville.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   │   ├── USGS_gage_event_runoff_lwi-transition-zone.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   │   ├── USGS_gage_hydro_variables_jacksonville.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   │   ├── USGS_gage_hydro_variables_lwi-transition-zone.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   │   ├── USGS_gage_hydro_variables_w_year_jacksonville.csv  #output from running the Model_Continuous Paramter_Data Notebook
    │   │   ├── USGS_gage_rain_stats_jacksonville.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   │   └── USGS_gage_rain_stats_lwi-transition-zone.csv #output from running the Model_Continuous Paramter_Data Notebook
    │   └── figures/
    |       ├── Figures 3, 5, 6, 7.nb
    │       ├── Figure_4_AND_C1.nb
    │       ├── Figure_8.nb
    │       ├── Figure_9.nb
    |       ├── Figures 10, 11.ipynb
    |       ├── Figure 12.nb
    │       ├── Figures_13_AND_14.nb
    │       └── Figures_15_AND_16.nb
    └── src/                        # Source code for data access, processing, and utilities
        └── data/
            ├── __init__.py
            ├── noaa_datasets.py       # Methods for handling NOAA climate datasets
            ├── noaa_ftp.py            # FTP handling for NOAA data
            ├── utils.py               # General utility functions
            ├── utils_ET.py            # Functions for handling evapotranspiration data
            ├── utils_files.py         # File parsing and management tools
            ├── utils_geo.py           # GIS utilities for spatial data
            └── utils_statistics.y     # Statistical analysis functions
```

---

## 🚀 Key Features

While established semi-distributed models are foundational for representing spatial heterogeneity and upscaled watershed dynamics, they have not been integrated with stochastic ecohydrology or extended to yield analytical, probabilistic descriptions of watershed-scale soil moisture and fluxes. This model addresses that gap by providing a stochastic ecohydrological perspective on semi-distributed rainfall–runoff dynamics, unifying three previously distinct modeling paradigms:
- **Semi-distributed heterogenity structure** The model incorporates both multiple conceptual soil layers and spatial heterogeneity from semi-distributed modeling. This spatial heterogeneity is defined either implicitly by PDFs or explicitly through indices (such as the topographic wetness index) calculated at each watershed point based on watershed attributes.  Based on this description of spatial heterogeneity, we formalize point-process upscaling using a mean-field approximation from statistical physics..
- **SCS-CN runof curve integration** The model adopts an extended version of the SCS-CN method (called the SCS-CNx method) as the semi-distributed component of the framework. The underlying spatial heterogeneity of the SCS-CN method then directly links point-scale processes to upscaled (unit-area) counterparts. This upscaling produces both the SCS-CN and SCS-CNx rainfall-runoff curves and yields upscaled baseflow and evapotranspiration fluxes consistent with the implicit spatial structure of the SCS-CN method. These upscaled fluxes are then coupled with stochastic ecohydrological modeling..
- **Stochastic ecohydrological coupling** Point-scale processes (e.g., evapotranspiration and soil moisture dynamics) are described using minimalist stochastic ecohydrology formulations. These are upscaled based on the spatial heterogeneity assumed by the SCS-CN method. This coupling enables continuous interstorm ecohydrological processes to interact with storm-event rainfall-runoff processes within a unified watershed-scale analytical framework. Techniques from stochastic ecohydrology are used to derive analytical PDFs for watershed state variables and fluxes. Ecohydrological relationships---such as watershed scale Budyko-type curves---emerge naturally as internal outcomes of the coupled system, rather than being imposed externally.

As a result, the framework links SCS-CN runoff generation to fundamental stochastic ecohydrological variables such as PET, LAI, plant wilting point, the Budyko dryness index, storm intermittency, and the storage index (effective soil depth over the average rainfall per storm event). These variables emerge from the merged semi-distributed, SCS-CN, and ecohydrological model structure, supporting physically interpretable upscaling of stochastic ecohydrology to the watershed scale.

---

## 📦 Requirements

The code is modular and can be run with standard Python tools and packages. A `requirements.txt` file is recommended but not included in the current ZIP. Key packages likely needed:

- `numpy`
- `pandas`
- `xarray`
- `matplotlib`
- `rasterio`
- `geopandas`
- `scipy`

For MODIS and DayMet scripts, API access or local data downloads may be required.

---

## 📝 Usage

1. Clone or download the repository.
2. Install dependencies.
3. Run Jupyter and Mathematica notebooks in `notebooks/` or Python scripts to:
   - Preprocess rainfall and ET data
   - Calibrate model parameters

---

## 📚 Citation

If you use this code in a publication, please cite:

Bartlett, M. S., Cultra, E., Geldner, N., & Porporato, A. (2025). *Stochastic ecohydrological perspective on semi-distributed rainfall–runoff dynamics*.

For data citation:
- USGS Water Data for the Nation:  
  U.S. Geological Survey, 2016, [DOI: 10.5066/F7P55KJN](https://doi.org/10.5066/F7P55KJN)
- MODIS ET:  
  Running, S. et al. (2021), [DOI: 10.5067/MODIS/MOD16A3GF.061](https://doi.org/10.5067/MODIS/MOD16A3GF.061)

---

## 📬 Contact

For questions, contact Mark S. Bartlett.
