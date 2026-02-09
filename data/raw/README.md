# Datasets Documentation

## 1. Epiclim_Final_data.csv (Primary)

**Source:** EpiClim Database (India epidemiological surveillance + climate)  
**URL:** https://www.epiclim.org/  
**Downloaded:** Pre-existing  
**License:** Open (check EpiClim website)  

### Schema
| Column | Type | Description |
|--------|------|-------------|
| week_of_outbreak | string | Week ("1st week", "2nd week", etc.) |
| state_ut | string | State/UT name |
| district | string | District name |
| Disease | string | Disease name (filter: "Chikungunya") |
| Cases | object | Case count (needs type conversion) |
| Deaths | float | Death count (many NaN) |
| day, mon, year | int | Date components |
| Latitude, Longitude | float | District centroid coordinates |
| preci | float | Precipitation (mm) - 136 NaN |
| LAI | float | Leaf Area Index - 2195 NaN |
| Temp | float | Temperature (Kelvin!) - 938 NaN |

### Stats
- **Rows:** 8,985 total
- **Chikungunya rows:** 731
- **Year range:** 2009–2022
- **States with Chikungunya:** 21

---

## 2. india_census_2011_district.csv

**Source:** India Census 2011 via Kaggle/GitHub  
**URL:** https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/master/india-districts-census-2011.csv  
**Downloaded:** 2026-01-26  
**License:** Public domain (Census data)  

### Key Columns
| Column | Type | Description |
|--------|------|-------------|
| District code | int | Census district code |
| State name | string | State name (UPPERCASE) |
| District name | string | District name |
| Population | int | Total population (2011 Census) |
| Male, Female | int | Gender breakdown |
| Literate | int | Literate population |
| SC, ST | int | Scheduled Caste/Tribe |
| + 110 more columns | various | Detailed demographics |

### Stats
- **Districts:** 640
- **States/UTs:** 35
- **Year:** 2011 Census

### Note
Census 2011 is the latest official census. India's 2021 census was delayed due to COVID-19. For 2025 projections, apply state-level growth rates from official Technical Group reports.

---

## Data Processing Notes

1. **Temperature:** Convert Kelvin → Celsius (subtract 273.15)
2. **Week parsing:** "1st week" → 1, "2nd week" → 2, etc.
3. **District matching:** Use fuzzy matching between datasets (different naming conventions)
4. **Population projection:** Apply ~2% annual growth rate for 2011→2025 estimates if needed
