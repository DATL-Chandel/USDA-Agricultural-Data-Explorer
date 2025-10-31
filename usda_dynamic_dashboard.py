"""
USDA QuickStats Dynamic Dashboard
==================================
Progressive disclosure interface with cascading dropdowns.
Each selection filters the next set of options dynamically.

Author: Agricultural Data Explorer
Version: 2.0 (Dynamic)
"""

import streamlit as st
import pandas as pd
import requests
import altair as alt
import io
import os
import time
from typing import List, Dict, Optional, Tuple
# from dotenv import load_dotenv, find_dotenv  # Commented for Streamlit Cloud deployment

# =============================================================================
# CONFIGURATION
# =============================================================================

# For local development: Uncomment the lines below and create a .env file with USDA_API_KEY
# Load environment variables from a local .env file if present
# load_dotenv(find_dotenv(), override=False)

# Resolve API key: Try Streamlit secrets first (for deployment), then environment variable (local or Cloud)
API_KEY = None
try:
    # Streamlit Cloud secrets (preferred for deployment)
    API_KEY = st.secrets.get("USDA_API_KEY", None)
except Exception:
    pass

# Fallback to environment variable (works for local .env or Streamlit Cloud env vars)
if not API_KEY:
    API_KEY = os.getenv("USDA_API_KEY")
BASE_URL = "https://quickstats.nass.usda.gov/api"

# =============================================================================
# COLUMN METADATA - Dynamic Column Management
# =============================================================================

COLUMN_METADATA = {
    # Essential columns (always important)
    'essential': {
        'year': {'display': 'Year', 'tooltip': 'Year of data collection', 'format': 'd'},
        'Value': {'display': 'Value', 'tooltip': 'Primary measurement value', 'format': ',.0f'},
        'state_name': {'display': 'State', 'tooltip': 'State name'},
        'county_name': {'display': 'County', 'tooltip': 'County name'},
        'region_desc': {'display': 'Region', 'tooltip': 'Geographic region'},
        'asd_desc': {'display': 'Agricultural District', 'tooltip': 'Ag district'},
    },
    
    # Contextual columns (show based on selection)
    'contextual': {
        'commodity_desc': {'display': 'Commodity', 'tooltip': 'Agricultural product'},
        'statisticcat_desc': {'display': 'Statistic Type', 'tooltip': 'What is being measured'},
        'unit_desc': {'display': 'Unit', 'tooltip': 'Measurement unit'},
        'short_desc': {'display': 'Description', 'tooltip': 'Full description'},
        'class_desc': {'display': 'Class', 'tooltip': 'Data classification'},
    },
    
    # Technical columns (useful but not always needed)
    'technical': {
        'CV (%)': {'display': 'Data Reliability (%)', 'tooltip': 'Coefficient of Variation - Lower is better (<15% is very reliable)', 'format': '.1f'},
        'prodn_practice_desc': {'display': 'Production Practice', 'tooltip': 'Farming method'},
        'util_practice_desc': {'display': 'Utilization Practice', 'tooltip': 'Usage method'},
        'freq_desc': {'display': 'Frequency', 'tooltip': 'Measurement frequency'},
    },
    
    # Junk columns (always hide - no value to users)
    'hidden': [
        'begin_code', 'end_code', 'reference_period_desc', 'week_ending', 
        'load_time', 'zip_5', 'watershed_code', 'watershed_desc',
        'congr_district_code', 'asd_code', 'county_code', 'state_fips_code',
        'country_code', 'location_desc', 'state_alpha', 'agg_level_desc',
        'domain_desc', 'domaincat_desc', 'sector_desc', 'group_desc',
        'source_desc'
    ]
}


def get_display_columns(data, agg_level, selected_commodity, selected_statistics):
    """
    Smart column selection based on context.
    Returns list of column names to display.
    """
    display_cols = []
    
    # 1. ADD LOCATION COLUMNS (based on aggregation level)
    if agg_level == "NATIONAL":
        pass  # No location column needed
    elif "REGION" in agg_level:
        if 'region_desc' in data.columns:
            display_cols.append('region_desc')
    elif "DISTRICT" in agg_level:
        if 'state_name' in data.columns:
            display_cols.append('state_name')
        if 'asd_desc' in data.columns:
            display_cols.append('asd_desc')
    elif agg_level == "COUNTY":
        if 'state_name' in data.columns:
            display_cols.append('state_name')
        if 'county_name' in data.columns:
            display_cols.append('county_name')
    elif agg_level == "STATE":
        if 'state_name' in data.columns:
            display_cols.append('state_name')
    
    # 2. ADD YEAR (always)
    if 'year' in data.columns:
        display_cols.append('year')
    
    # 3. ADD COMMODITY (only if multiple or user needs context)
    if 'commodity_desc' in data.columns:
        if data['commodity_desc'].nunique() > 1:
            display_cols.append('commodity_desc')
    
    # 4. ADD STATISTIC TYPE (only if multiple)
    if 'statisticcat_desc' in data.columns:
        if len(selected_statistics) > 1 or data['statisticcat_desc'].nunique() > 1:
            display_cols.append('statisticcat_desc')
    
    # 5. ADD VALUE (always)
    if 'Value' in data.columns:
        display_cols.append('Value')
    
    # 6. ADD UNIT (only if it varies)
    if 'unit_desc' in data.columns:
        if data['unit_desc'].nunique() > 1:
            display_cols.append('unit_desc')
    
    # 7. ADD RELIABILITY (if available and has data)
    if 'CV (%)' in data.columns:
        if data['CV (%)'].notna().sum() > 0:
            display_cols.append('CV (%)')
    
    # 8. ADD CLASS/DESCRIPTION (if useful)
    if 'class_desc' in data.columns:
        if data['class_desc'].notna().sum() > 0 and data['class_desc'].nunique() > 1:
            display_cols.append('class_desc')
    
    return display_cols


def get_friendly_column_names(selected_commodity, selected_statistics):
    """
    Get context-aware friendly column names.
    Returns dictionary for renaming.
    """
    friendly_names = {}
    
    # Build from metadata
    for category in ['essential', 'contextual', 'technical']:
        for col, info in COLUMN_METADATA[category].items():
            friendly_names[col] = info['display']
    
    # DYNAMIC VALUE COLUMN NAME
    if len(selected_statistics) == 1:
        stat = selected_statistics[0]
        # Make it more descriptive
        if 'EXPENSE' in stat.upper() or 'COST' in stat.upper():
            friendly_names['Value'] = f"{selected_commodity.title()} {stat.title()} ($)"
        elif 'PRODUCTION' in stat.upper():
            friendly_names['Value'] = f"{selected_commodity.title()} {stat.title()}"
        elif 'YIELD' in stat.upper():
            friendly_names['Value'] = f"{selected_commodity.title()} {stat.title()}"
        elif 'PRICE' in stat.upper():
            friendly_names['Value'] = f"{selected_commodity.title()} {stat.title()} ($)"
        else:
            friendly_names['Value'] = f"{selected_commodity.title()} - {stat.title()}"
    
    return friendly_names


def clean_dataframe_for_display(data, display_columns, friendly_names):
    """
    Clean and prepare dataframe for user display.
    """
    # Select only display columns that exist
    existing_cols = [col for col in display_columns if col in data.columns]
    clean_data = data[existing_cols].copy()
    
    # Rename columns
    rename_dict = {col: friendly_names.get(col, col) for col in existing_cols}
    clean_data = clean_data.rename(columns=rename_dict)
    
    return clean_data

# Page config
st.set_page_config(
    page_title="USDA Dynamic Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS - API CALLS
# =============================================================================

def make_request(url: str, params: Dict[str, str], retries: int = 3, backoff: float = 1.5) -> Optional[requests.Response]:
    """HTTP GET with simple retry/backoff for 429/5xx/timeouts."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            # Retry on rate limiting or server errors
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < retries - 1:
                    time.sleep(backoff ** attempt)
                    continue
            return response
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
                continue
    if last_exc is not None:
        st.error(f"Network error contacting USDA API: {last_exc}")
    return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_all_commodities() -> List[str]:
    """Fetch complete list of commodities from API."""
    url = f"{BASE_URL}/get_param_values/"
    params = {"key": API_KEY, "param": "commodity_desc"}
    
    try:
        response = make_request(url, params)
        if response and response.status_code == 200:
            data = response.json()
            return sorted(data.get("commodity_desc", []))
        elif response is not None:
            st.warning(f"USDA API returned status {response.status_code} when fetching commodities.")
        return []
    except Exception as e:
        st.error(f"Error fetching commodities: {e}")
        return []


@st.cache_data(ttl=86400)
def categorize_commodities(commodities: List[str]) -> Dict[str, List[str]]:
    """Group commodities into logical categories."""
    
    # Define category keywords for classification
    categories = {
        "Field Crops": [
            "CORN", "SOYBEANS", "WHEAT", "COTTON", "SORGHUM", "BARLEY", 
            "OATS", "RYE", "RICE", "CANOLA", "SUNFLOWER", "PEANUTS"
        ],
        "Fruits": [
            "APPLES", "ORANGES", "GRAPES", "STRAWBERRIES", "BLUEBERRIES",
            "PEACHES", "CHERRIES", "PEARS", "PLUMS", "RASPBERRIES"
        ],
        "Vegetables": [
            "TOMATOES", "POTATOES", "ONIONS", "LETTUCE", "CABBAGE",
            "CARROTS", "PEPPERS", "CUCUMBERS", "SQUASH", "BROCCOLI"
        ],
        "Nuts & Seeds": [
            "ALMONDS", "WALNUTS", "PECANS", "PISTACHIOS", "HAZELNUTS"
        ],
        "Animals": [
            "CATTLE", "HOGS", "SHEEP", "GOATS", "CHICKENS", "TURKEYS"
        ],
        "Animal Products": [
            "MILK", "EGGS", "HONEY", "WOOL"
        ],
        "Specialty Crops": [
            "HOPS", "TOBACCO", "SUGARCANE", "SUGAR BEETS", "MINT"
        ],
        "Economics & Expenses": [
            "FERTILIZER", "CHEMICAL", "LABOR", "FUEL", "SEED", "RENT",
            "EXPENSE", "INCOME", "AG LAND", "ASSET", "INTEREST",
            "CUSTOM WORK", "UTILITIES", "OPERATING"
        ],
        "Environmental Data": [
            "APPLICATIONS", "TREATED", "CONSERVATION", "IRRIGATION",
            "PRACTICES", "NUTRIENT"
        ],
        "Farm Operations & Demographics": [
            "FARM OPERATIONS", "OPERATORS", "FARMS", "PRODUCERS"
        ],
        "Other": []
    }
    
    # Categorize each commodity
    categorized = {cat: [] for cat in categories.keys()}
    
    for commodity in commodities:
        assigned = False
        for category, keywords in categories.items():
            if any(keyword in commodity.upper() for keyword in keywords):
                categorized[category].append(commodity)
                assigned = True
                break
        
        if not assigned:
            categorized["Other"].append(commodity)
    
    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_param_values(param_name: str, **filters) -> List[str]:
    """Generic function to get any parameter values with optional filters."""
    url = f"{BASE_URL}/get_param_values/"
    params = {"key": API_KEY, "param": param_name}
    params.update(filters)
    
    try:
        response = make_request(url, params)
        if response and response.status_code == 200:
            data = response.json()
            values = data.get(param_name, [])
            return sorted(values) if values else []
        elif response is not None:
            st.warning(f"USDA API returned status {response.status_code} for {param_name} with filters {filters}.")
        return []
    except Exception as e:
        st.error(f"Error fetching {param_name}: {e}")
        return []


@st.cache_data(ttl=3600)
def get_available_statistics(commodity: str) -> List[str]:
    """Get statistics available for a specific commodity."""
    return get_param_values("statisticcat_desc", commodity_desc=commodity)


@st.cache_data(ttl=3600)
def get_available_agg_levels(commodity: str, statistics: List[str]) -> List[str]:
    """Get aggregation levels available for commodity + statistics combination."""
    all_levels = set()
    
    for stat in statistics:
        levels = get_param_values(
            "agg_level_desc",
            commodity_desc=commodity,
            statisticcat_desc=stat
        )
        all_levels.update(levels)
    
    return sorted(list(all_levels))


@st.cache_data(ttl=3600)
def get_available_states(commodity: str, statistics: List[str], agg_level: str) -> List[str]:
    """Get states available for commodity + statistics + agg_level."""
    return get_param_values(
        "state_name",
        commodity_desc=commodity,
        statisticcat_desc=statistics[0],
        agg_level_desc=agg_level
    )


@st.cache_data(ttl=3600)
def get_available_regions(commodity: str, statistics: List[str], agg_level: str = None) -> List[str]:
    """Get regions available for commodity + statistics."""
    # Use provided agg_level or default to "REGION"
    agg_level_param = agg_level if agg_level else "REGION"
    return get_param_values(
        "region_desc",
        commodity_desc=commodity,
        statisticcat_desc=statistics[0],
        agg_level_desc=agg_level_param
    )


@st.cache_data(ttl=3600)
def get_available_districts(commodity: str, statistics: List[str], state: str) -> List[str]:
    """Get ag districts available for state."""
    return get_param_values(
        "asd_desc",
        commodity_desc=commodity,
        statisticcat_desc=statistics[0],
        state_name=state,
        agg_level_desc="AG DISTRICT"
    )


@st.cache_data(ttl=3600)
def get_available_counties(commodity: str, statistics: List[str], state: str) -> List[str]:
    """Get counties available for state."""
    return get_param_values(
        "county_name",
        commodity_desc=commodity,
        statisticcat_desc=statistics[0],
        state_name=state,
        agg_level_desc="COUNTY"
    )


@st.cache_data(ttl=3600)
def get_available_years(commodity: str, statistics: List[str], agg_level: str, 
                        locations: List[str]) -> List[int]:
    """Get available years for selected parameters."""
    # Determine the correct location parameter based on agg_level
    params = {
        "commodity_desc": commodity,
        "statisticcat_desc": statistics[0],
        "agg_level_desc": agg_level
    }
    
    # Add location parameter based on agg_level
    if locations and locations[0] != "US TOTAL":
        if "REGION" in agg_level:
            params["region_desc"] = locations[0]
        elif "DISTRICT" in agg_level:
            params["asd_desc"] = locations[0]
        elif "COUNTY" in agg_level:
            params["county_name"] = locations[0]
        elif agg_level == "STATE":
            params["state_name"] = locations[0]
    
    years = get_param_values("year", **params)
    
    valid_years = sorted([int(y) for y in years if y.isdigit()], reverse=True)
    return valid_years


def fetch_final_data(commodity: str, statistics: List[str], agg_level: str, 
                     locations: List[str], selected_years: List[int]) -> pd.DataFrame:
    """Fetch the actual data based on all selections."""
    url = f"{BASE_URL}/api_GET/"
    
    # Determine the correct location parameter based on agg_level
    # Use flexible matching for variations like "REGION : MULTI-STATE"
    if agg_level == "NATIONAL":
        location_param = None
    elif "REGION" in agg_level:
        location_param = "region_desc"
    elif "DISTRICT" in agg_level:
        location_param = "asd_desc"
    elif "COUNTY" in agg_level:
        location_param = "county_name"
    elif agg_level == "STATE":
        location_param = "state_name"
    else:
        # Default fallback
        location_param = "state_name"
    
    all_dfs = []

    @st.cache_data(ttl=3600)
    def fetch_csv_for(
        commodity_local: str,
        stat_local: str,
        agg_level_local: str,
        location_param_local: Optional[str],
        location_local: str,
        year_ge: Optional[int],
        year_le: Optional[int],
        api_key_local: Optional[str]
    ) -> pd.DataFrame:
        params_local = {
            "key": api_key_local,
            "format": "CSV",
            "commodity_desc": commodity_local,
            "statisticcat_desc": stat_local,
            "agg_level_desc": agg_level_local
        }
        if location_param_local and location_local != "US TOTAL":
            params_local[location_param_local] = location_local
        if year_ge is not None and year_le is not None:
            params_local["year__GE"] = str(year_ge)
            params_local["year__LE"] = str(year_le)
        response_local = make_request(url, params_local)
        if response_local and response_local.status_code == 200:
            try:
                return pd.read_csv(io.StringIO(response_local.text), low_memory=False)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    
    for stat in statistics:
        for location in locations:
            df = fetch_csv_for(
                commodity,
                stat,
                agg_level,
                location_param,
                location,
                min(selected_years) if selected_years else None,
                max(selected_years) if selected_years else None,
                API_KEY
            )
            if not df.empty:
                all_dfs.append(df)
            else:
                st.warning(f"No data returned for {stat} in {location}.")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Clean data
        combined['Value'] = pd.to_numeric(combined['Value'].astype(str).str.replace(',', ''), errors='coerce')
        combined = combined.dropna(subset=['Value'])
        combined['year'] = pd.to_numeric(combined['year'], errors='coerce')
        
        # Fix mixed-type columns (especially CV (%))
        # Convert all object columns that should be numeric
        for col in combined.columns:
            if combined[col].dtype == 'object':
                # Try to convert to numeric, keep as string if fails
                try:
                    # Remove commas and convert
                    converted = pd.to_numeric(combined[col].astype(str).str.replace(',', ''), errors='coerce')
                    # If more than 50% converted successfully, use numeric
                    if converted.notna().sum() / len(converted) > 0.5:
                        combined[col] = converted
                    else:
                        # Keep as string but clean up 'nan' strings and empty values
                        combined[col] = combined[col].astype(str)
                        combined[col] = combined[col].replace(['nan', 'None', ''], pd.NA)
                except:
                    # Keep as string but clean up 'nan' strings
                    combined[col] = combined[col].astype(str)
                    combined[col] = combined[col].replace(['nan', 'None', ''], pd.NA)
        
        # Drop rows with missing location columns (critical for charts)
        location_cols = ['state_name', 'region_desc', 'county_name', 'asd_desc']
        existing_location_cols = [col for col in location_cols if col in combined.columns]
        if existing_location_cols:
            # At least one location column must have valid data
            combined = combined.dropna(subset=existing_location_cols, how='all')
        
        return combined
    
    return pd.DataFrame()


# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if 'step1_complete' not in st.session_state:
        st.session_state.step1_complete = False
        st.session_state.step2_complete = False
        st.session_state.step3_complete = False
        st.session_state.step4_complete = False
        st.session_state.step5_complete = False
        st.session_state.step6_complete = False
        
        st.session_state.selected_category = None
        st.session_state.selected_commodity = None
        st.session_state.selected_statistics = []
        st.session_state.selected_agg_level = None
        st.session_state.selected_state = None
        st.session_state.selected_locations = []
        st.session_state.selected_years = []
        
        st.session_state.data_loaded = False
        st.session_state.final_data = None


def reset_from_step(step_number: int):
    """Reset all steps from the given step onwards."""
    if step_number <= 1:
        st.session_state.step1_complete = False
        st.session_state.selected_category = None
    if step_number <= 2:
        st.session_state.step2_complete = False
        st.session_state.selected_commodity = None
    if step_number <= 3:
        st.session_state.step3_complete = False
        st.session_state.selected_statistics = []
    if step_number <= 4:
        st.session_state.step4_complete = False
        st.session_state.selected_agg_level = None
    if step_number <= 5:
        st.session_state.step5_complete = False
        st.session_state.selected_state = None
        st.session_state.selected_locations = []
    if step_number <= 6:
        st.session_state.step6_complete = False
        st.session_state.selected_years = []
        st.session_state.data_loaded = False
        if 'temp_years' in st.session_state:
            del st.session_state.temp_years


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application logic."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("USDA Agricultural Data Explorer")
    st.markdown("### Dynamic Dashboard with Smart Filtering")
    st.markdown("---")

    # Validate API key
    if not API_KEY:
        st.error("USDA_API_KEY environment variable is not set. Please set USDA_API_KEY and restart the app.")
        return
    
    # Create two columns
    col_sidebar, col_main = st.columns([3, 7])
    
    # =============================================================================
    # SIDEBAR - Progressive Steps
    # =============================================================================
    
    with col_sidebar:
        st.markdown("### Build Your Dashboard")
        st.markdown("Complete each step to unlock the next")
        st.markdown("---")
        
        # Get all commodities once
        all_commodities = get_all_commodities()
        if not all_commodities:
            st.error("Could not fetch commodity list. Please check your API key.")
            return
        
        categorized = categorize_commodities(all_commodities)
        
        # =====================================================================
        # STEP 1: Select Category
        # =====================================================================
        
        st.markdown("#### Step 1: Select Category")
        
        if st.session_state.step1_complete:
            st.success(f"✓ Selected: {st.session_state.selected_category}")
            if st.button("Change Category", key="change_cat"):
                reset_from_step(1)
                st.rerun()
        else:
            category = st.selectbox(
                "Choose a commodity category:",
                options=[""] + list(categorized.keys()),
                key="category_selector"
            )
            
            if category:
                st.session_state.selected_category = category
                st.session_state.step1_complete = True
                st.rerun()
        
        # =====================================================================
        # STEP 2: Select Commodity
        # =====================================================================
        
        if st.session_state.step1_complete:
            st.markdown("---")
            st.markdown("#### Step 2: Select Commodity")
            
            if st.session_state.step2_complete:
                st.success(f"✓ Selected: {st.session_state.selected_commodity}")
                if st.button("Change Commodity", key="change_comm"):
                    reset_from_step(2)
                    st.rerun()
            else:
                commodities_in_category = categorized[st.session_state.selected_category]
                commodity = st.selectbox(
                    f"Choose from {st.session_state.selected_category}:",
                    options=[""] + commodities_in_category,
                    key="commodity_selector"
                )
                
                if commodity:
                    st.session_state.selected_commodity = commodity
                    st.session_state.step2_complete = True
                    st.rerun()
        
        # =====================================================================
        # STEP 3: Select Statistics
        # =====================================================================
        
        if st.session_state.step2_complete:
            st.markdown("---")
            st.markdown("#### Step 3: Select Statistics")
            
            if st.session_state.step3_complete:
                stats_str = ", ".join(st.session_state.selected_statistics)
                st.success(f"✓ Selected: {stats_str}")
                if st.button("Modify Statistics", key="change_stats"):
                    reset_from_step(3)
                    st.rerun()
            else:
                with st.spinner("Checking available statistics..."):
                    available_stats = get_available_statistics(st.session_state.selected_commodity)
                
                if available_stats:
                    st.info(f"Select one or more statistics for {st.session_state.selected_commodity}")
                    
                    selected_stats = []
                    for stat in available_stats:
                        # Display statistics without record counts for speed
                        if st.checkbox(stat, key=f"stat_{stat}"):
                            selected_stats.append(stat)
                    
                    if selected_stats:
                        if st.button("Continue with Selected Statistics", type="primary"):
                            st.session_state.selected_statistics = selected_stats
                            st.session_state.step3_complete = True
                            st.rerun()
                else:
                    st.warning("No statistics available for this commodity")
        
        # =====================================================================
        # STEP 4: Select Aggregation Level
        # =====================================================================
        
        if st.session_state.step3_complete:
            st.markdown("---")
            st.markdown("#### Step 4: Select Geographic Level")
            
            if st.session_state.step4_complete:
                st.success(f"✓ Selected: {st.session_state.selected_agg_level}")
                if st.button("Change Level", key="change_agg"):
                    reset_from_step(4)
                    st.rerun()
            else:
                with st.spinner("Loading geographic levels..."):
                    agg_levels = get_available_agg_levels(
                        st.session_state.selected_commodity,
                        st.session_state.selected_statistics
                    )
                
                if agg_levels:
                    agg_level = st.selectbox(
                        "Choose geographic level:",
                        options=[""] + agg_levels,
                        key="agg_selector"
                    )
                    
                    if agg_level:
                        st.session_state.selected_agg_level = agg_level
                        st.session_state.step4_complete = True
                        st.rerun()
                else:
                    st.warning("No geographic levels available")
        
        # =====================================================================
        # STEP 5: Select Locations
        # =====================================================================
        
        if st.session_state.step4_complete:
            st.markdown("---")
            st.markdown("#### Step 5: Select Location(s)")
            
            if st.session_state.step5_complete:
                locs_str = ", ".join(st.session_state.selected_locations)
                st.success(f"✓ Selected: {locs_str}")
                if st.button("Modify Locations", key="change_locs"):
                    reset_from_step(5)
                    st.rerun()
            else:
                agg_level = st.session_state.selected_agg_level
                
                # NATIONAL level - no location selection needed
                if agg_level == "NATIONAL":
                    st.info("National level - no location selection needed")
                    if st.button("Continue", type="primary"):
                        st.session_state.selected_locations = ["US TOTAL"]
                        st.session_state.step5_complete = True
                        st.rerun()
                
                # REGION level - select regions directly (handles "REGION" or "REGION : MULTI-STATE")
                elif "REGION" in agg_level:
                    with st.spinner("Loading regions..."):
                        regions = get_available_regions(
                            st.session_state.selected_commodity,
                            st.session_state.selected_statistics,
                            agg_level
                        )
                    
                    if regions:
                        selected_locs = st.multiselect(
                            "Select region(s):",
                            options=regions,
                            key="region_selector"
                        )
                        
                        if selected_locs:
                            if st.button("Continue", type="primary"):
                                st.session_state.selected_locations = selected_locs
                                st.session_state.step5_complete = True
                                st.rerun()
                    else:
                        st.warning("No regions available")
                
                # STATE level - select states
                elif agg_level == "STATE":
                    with st.spinner("Loading states..."):
                        states = get_available_states(
                            st.session_state.selected_commodity,
                            st.session_state.selected_statistics,
                            st.session_state.selected_agg_level
                        )
                    
                    if states:
                        selected_locs = st.multiselect(
                            "Select state(s):",
                            options=states,
                            key="state_selector"
                        )
                        
                        if selected_locs:
                            if st.button("Continue", type="primary"):
                                st.session_state.selected_locations = selected_locs
                                st.session_state.step5_complete = True
                                st.rerun()
                    else:
                        st.warning("No states available")
                
                # AG DISTRICT level - select state first, then districts
                elif agg_level == "AG DISTRICT":
                    if not st.session_state.selected_state:
                        with st.spinner("Loading states..."):
                            states = get_available_states(
                                st.session_state.selected_commodity,
                                st.session_state.selected_statistics,
                                "AG DISTRICT"
                            )
                        
                        if states:
                            state = st.selectbox(
                                "First, select a state:",
                                options=[""] + states,
                                key="state_for_district"
                            )
                            
                            if state:
                                st.session_state.selected_state = state
                                st.rerun()
                        else:
                            st.warning("No states with district-level data")
                    else:
                        st.info(f"State: {st.session_state.selected_state}")
                        
                        with st.spinner("Loading districts..."):
                            districts = get_available_districts(
                                st.session_state.selected_commodity,
                                st.session_state.selected_statistics,
                                st.session_state.selected_state
                            )
                        
                        if districts:
                            selected_locs = st.multiselect(
                                "Select district(s):",
                                options=districts,
                                key="district_selector"
                            )
                            
                            if selected_locs:
                                if st.button("Continue", type="primary"):
                                    st.session_state.selected_locations = selected_locs
                                    st.session_state.step5_complete = True
                                    st.rerun()
                        else:
                            st.warning("No districts available for this state")
                
                # COUNTY level - select state first, then counties
                elif agg_level == "COUNTY":
                    if not st.session_state.selected_state:
                        with st.spinner("Loading states..."):
                            states = get_available_states(
                                st.session_state.selected_commodity,
                                st.session_state.selected_statistics,
                                "COUNTY"
                            )
                        
                        if states:
                            state = st.selectbox(
                                "First, select a state:",
                                options=[""] + states,
                                key="state_for_county"
                            )
                            
                            if state:
                                st.session_state.selected_state = state
                                st.rerun()
                        else:
                            st.warning("No states with county-level data")
                    else:
                        st.info(f"State: {st.session_state.selected_state}")
                        
                        with st.spinner("Loading counties..."):
                            counties = get_available_counties(
                                st.session_state.selected_commodity,
                                st.session_state.selected_statistics,
                                st.session_state.selected_state
                            )
                        
                        if counties:
                            # Add search box
                            search = st.text_input("Search counties:", key="county_search")
                            
                            filtered_counties = counties
                            if search:
                                filtered_counties = [c for c in counties if search.upper() in c.upper()]
                            
                            selected_locs = st.multiselect(
                                f"Select county(ies) - {len(filtered_counties)} available:",
                                options=filtered_counties,
                                key="county_selector"
                            )
                            
                            if selected_locs:
                                if st.button("Continue", type="primary"):
                                    st.session_state.selected_locations = selected_locs
                                    st.session_state.step5_complete = True
                                    st.rerun()
                        else:
                            st.warning("No counties available for this state")
                
                else:
                    st.warning(f"Location selection not implemented for {agg_level}")
        
        # =====================================================================
        # STEP 6: Select Time Period
        # =====================================================================
        
        if st.session_state.step5_complete:
            st.markdown("---")
            st.markdown("#### Step 6: Select Year(s)")
            
            if st.session_state.step6_complete:
                years_str = ", ".join([str(y) for y in sorted(st.session_state.selected_years, reverse=True)])
                st.success(f"✓ Selected: {years_str}")
                if st.button("Change Years", key="change_year"):
                    reset_from_step(6)
                    st.rerun()
            else:
                with st.spinner("Loading available years..."):
                    years_list = get_available_years(
                        st.session_state.selected_commodity,
                        st.session_state.selected_statistics,
                        st.session_state.selected_agg_level,
                        st.session_state.selected_locations
                    )
                
                if years_list:
                    st.info("Select one or more years")
                    
                    # Quick select buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Last 5 Years", width="stretch"):
                            st.session_state.temp_years = years_list[:5]
                    with col2:
                        if st.button("Last 10 Years", width="stretch"):
                            st.session_state.temp_years = years_list[:10]
                    with col3:
                        if st.button("All Years", width="stretch"):
                            st.session_state.temp_years = years_list
                    
                    # Use multiselect instead of checkboxes
                    default_selection = getattr(st.session_state, 'temp_years', [])
                    
                    selected_years = st.multiselect(
                        "Select year(s):",
                        options=years_list,
                        default=default_selection,
                        key="year_selector"
                    )
                    
                    if selected_years:
                        if st.button("Continue", type="primary"):
                            st.session_state.selected_years = selected_years
                            st.session_state.step6_complete = True
                            if 'temp_years' in st.session_state:
                                del st.session_state.temp_years
                            st.rerun()
                else:
                    st.warning("No years available for this combination")
        
        # =====================================================================
        # GET DATA BUTTON
        # =====================================================================
        
        if st.session_state.step6_complete and not st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### Ready to Load Data")
            
            if st.button("Get Data", type="primary", width="stretch"):
                with st.spinner("Loading data..."):
                    data = fetch_final_data(
                        st.session_state.selected_commodity,
                        st.session_state.selected_statistics,
                        st.session_state.selected_agg_level,
                        st.session_state.selected_locations,
                        st.session_state.selected_years
                    )
                    
                    if not data.empty:
                        st.session_state.final_data = data
                        st.session_state.data_loaded = True
                        st.success(f"Loaded {len(data):,} records.")
                        st.rerun()
                    else:
                        st.error("No data found for this combination.")
                        st.info("""**Suggestions:**
                        - Try expanding the date range (e.g., Last 5 Years instead of Last Year)
                        - Some commodities may not have all statistics at all levels
                        - County-level data may be limited compared to State-level
                        - Recent years may have incomplete data
                        """)
        
        # Progress indicator
        if any([st.session_state.step1_complete, st.session_state.step2_complete,
                st.session_state.step3_complete, st.session_state.step4_complete,
                st.session_state.step5_complete, st.session_state.step6_complete]):
            st.markdown("---")
            steps_complete = sum([
                st.session_state.step1_complete,
                st.session_state.step2_complete,
                st.session_state.step3_complete,
                st.session_state.step4_complete,
                st.session_state.step5_complete,
                st.session_state.step6_complete
            ])
            st.progress(steps_complete / 6.0)
            st.caption(f"Progress: {steps_complete}/6 steps complete")
            
            if st.button("↻ Start Over", width="stretch"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # =============================================================================
    # MAIN CONTENT AREA
    # =============================================================================
    
    with col_main:
        if st.session_state.data_loaded and st.session_state.final_data is not None:
            # Display data
            data = st.session_state.final_data
            
            st.markdown(f"## {st.session_state.selected_commodity} Analysis")
            st.markdown(f"**Locations:** {', '.join(st.session_state.selected_locations)}")
            st.markdown(f"**Statistics:** {', '.join(st.session_state.selected_statistics)}")
            years_str = ", ".join([str(y) for y in sorted(st.session_state.selected_years, reverse=True)])
            st.markdown(f"**Years:** {years_str}")
            
            # Get smart column display
            display_columns = get_display_columns(
                data, 
                st.session_state.selected_agg_level,
                st.session_state.selected_commodity,
                st.session_state.selected_statistics
            )
            friendly_names = get_friendly_column_names(
                st.session_state.selected_commodity,
                st.session_state.selected_statistics
            )
            
            st.markdown("---")
            
            # Summary statistics with better info
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Records", f"{len(data):,}")
            with col2:
                st.metric("Locations", len(st.session_state.selected_locations))
            with col3:
                st.metric("Statistics", len(st.session_state.selected_statistics))
            with col4:
                years = data['year'].nunique()
                st.metric("Years", years)
            with col5:
                st.metric("Columns Shown", f"{len(display_columns)} / {len(data.columns)}")
            
            # Info banner about smart display
            st.info(f"Smart Display: Showing {len(display_columns)} most relevant columns (hiding {len(data.columns) - len(display_columns)} unnecessary columns)")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Comparisons", "By Metric", "Data & Export"])
            
            with tab1:
                st.markdown("### Overview & Trends")
                
                # Enhanced summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                if 'Value' in data.columns:
                    with col1:
                        avg_value = data['Value'].mean()
                        st.metric("Average Value", f"{avg_value:,.1f}")
                    
                    with col2:
                        max_value = data['Value'].max()
                        max_loc = data[data['Value'] == max_value].iloc[0]
                        st.metric("Highest Value", f"{max_value:,.1f}",
                                 delta=f"{max_loc.get('state_name', 'N/A')}")
                    
                    with col3:
                        min_value = data['Value'].min()
                        st.metric("Lowest Value", f"{min_value:,.1f}")
                    
                    with col4:
                        if data['Value'].std() > 0:
                            cv = (data['Value'].std() / data['Value'].mean()) * 100
                            st.metric("Variability", f"{cv:.1f}%")
                
                st.markdown("---")
                
                # Main trend chart
                if 'year' in data.columns and 'Value' in data.columns:
                    # Determine grouping field - check all possible location columns
                    location_col = None
                    for col in ['state_name', 'region_desc', 'county_name', 'asd_desc']:
                        if col in data.columns:
                            location_col = col
                            break
                    
                    if location_col and len(st.session_state.selected_locations) > 1:
                        # Multi-location trend - filter out rows with missing location values
                        chart_data = data.dropna(subset=[location_col])
                        chart = alt.Chart(chart_data).mark_line(point=True, strokeWidth=3).encode(
                            x=alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y('Value:Q', title='Value', scale=alt.Scale(zero=False)),
                            color=alt.Color(f'{location_col}:N', title='Location', 
                                          legend=alt.Legend(orient='top')),
                            tooltip=['year', location_col, 'Value', 'statisticcat_desc']
                        ).properties(
                            height=400,
                            title=f"{st.session_state.selected_commodity} Trends by Location"
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        # Single location - show area chart
                        chart = alt.Chart(data).mark_area(
                            line={'color': '#2E7D32'},
                            color=alt.Gradient(
                                gradient='linear',
                                stops=[
                                    alt.GradientStop(color='#E8F5E9', offset=0),
                                    alt.GradientStop(color='#66BB6A', offset=1)
                                ],
                                x1=1, x2=1, y1=1, y2=0
                            )
                        ).encode(
                            x=alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y('Value:Q', title='Value'),
                            tooltip=['year', 'Value', 'statisticcat_desc']
                        ).properties(
                            height=400,
                            title=f"{st.session_state.selected_commodity} Trend"
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
            
            with tab2:
                st.markdown("### Location & Year Comparisons")
                
                # Determine location column - check all possible location columns
                location_col = None
                for col in ['state_name', 'region_desc', 'county_name', 'asd_desc']:
                    if col in data.columns and col in display_columns:
                        location_col = col
                        break
                
                if location_col and len(st.session_state.selected_locations) > 1:
                    # Latest year comparison across locations
                    st.markdown("#### Latest Year Comparison")
                    latest_year = data['year'].max()
                    latest_data = data[data['year'] == latest_year].copy()
                    # Filter out rows with missing location values
                    latest_data = latest_data.dropna(subset=[location_col])
                    
                    if not latest_data.empty:
                        # Aggregate by location and statistic
                        comparison_data = latest_data.groupby([location_col, 'statisticcat_desc'])['Value'].mean().reset_index()
                        
                        if len(st.session_state.selected_statistics) == 1:
                            # Single statistic - horizontal bar chart
                            chart = alt.Chart(comparison_data).mark_bar().encode(
                                y=alt.Y(f'{location_col}:N', title='Location', sort='-x'),
                                x=alt.X('Value:Q', title='Value'),
                                color=alt.Color(f'{location_col}:N', legend=None),
                                tooltip=[location_col, 'Value', 'statisticcat_desc']
                            ).properties(
                                height=max(300, len(comparison_data) * 40),
                                title=f"{st.session_state.selected_statistics[0]} in {latest_year}"
                            )
                            
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            # Multiple statistics - grouped bar chart
                            chart = alt.Chart(comparison_data).mark_bar().encode(
                                x=alt.X(f'{location_col}:N', title='Location', axis=alt.Axis(labelAngle=-45)),
                                y=alt.Y('Value:Q', title='Value'),
                                color=alt.Color('statisticcat_desc:N', title='Statistic'),
                                column=alt.Column('statisticcat_desc:N', title=''),
                                tooltip=[location_col, 'statisticcat_desc', 'Value']
                            ).properties(
                                width=200,
                                height=350
                            )
                            
                            st.altair_chart(chart)
                
                # Year-over-year change calculation
                if 'year' in data.columns and len(data['year'].unique()) > 1:
                    st.markdown("#### Year-over-Year Changes")
                    
                    years = sorted(data['year'].unique())
                    if len(years) >= 2:
                        latest_year = years[-1]
                        previous_year = years[-2]
                        
                        # Calculate changes - filter out rows with missing location values
                        if location_col:
                            clean_data = data.dropna(subset=[location_col])
                            latest = clean_data[clean_data['year'] == latest_year].groupby(location_col)['Value'].mean()
                            previous = clean_data[clean_data['year'] == previous_year].groupby(location_col)['Value'].mean()
                        else:
                            latest = None
                            previous = None
                        
                        if latest is not None and previous is not None and not latest.empty and not previous.empty:
                            change_pct = ((latest - previous) / previous * 100).reset_index()
                            change_pct.columns = ['Location', 'Change_Pct']
                            
                            # Color based on positive/negative
                            chart = alt.Chart(change_pct).mark_bar().encode(
                                y=alt.Y('Location:N', sort='-x', title=''),
                                x=alt.X('Change_Pct:Q', title='% Change'),
                                color=alt.condition(
                                    alt.datum.Change_Pct > 0,
                                    alt.value('#2E7D32'),  # Green for positive
                                    alt.value('#D32F2F')   # Red for negative
                                ),
                                tooltip=['Location', alt.Tooltip('Change_Pct:Q', format='.1f', title='% Change')]
                            ).properties(
                                height=max(250, len(change_pct) * 35),
                                title=f"Change from {previous_year} to {latest_year}"
                            )
                            
                            st.altair_chart(chart, use_container_width=True)
                            
                            # Show numerical summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_change = change_pct['Change_Pct'].mean()
                                st.metric("Average Change", f"{avg_change:+.1f}%")
                            with col2:
                                max_gain = change_pct['Change_Pct'].max()
                                st.metric("Biggest Gain", f"{max_gain:+.1f}%")
                            with col3:
                                max_loss = change_pct['Change_Pct'].min()
                                st.metric("Biggest Loss", f"{max_loss:+.1f}%")
            
            with tab3:
                st.markdown("### Analysis by Statistic")
                
                # If multiple statistics, create faceted charts
                if len(st.session_state.selected_statistics) > 1:
                    st.info(f"Comparing {len(st.session_state.selected_statistics)} different statistics")
                    
                    for stat in st.session_state.selected_statistics:
                        stat_data = data[data['statisticcat_desc'] == stat]
                        
                        if not stat_data.empty:
                            st.markdown(f"#### {stat}")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Trend chart for this statistic
                                location_col = None
                                for col in ['state_name', 'region_desc', 'county_name', 'asd_desc']:
                                    if col in stat_data.columns:
                                        location_col = col
                                        break
                                
                                if location_col:
                                    # Filter out rows with missing location values
                                    chart_data = stat_data.dropna(subset=[location_col])
                                    chart = alt.Chart(chart_data).mark_line(point=True).encode(
                                        x=alt.X('year:O', title='Year'),
                                        y=alt.Y('Value:Q', title=stat, scale=alt.Scale(zero=False)),
                                        color=alt.Color(f'{location_col}:N', title='Location'),
                                        tooltip=['year', location_col, 'Value']
                                    ).properties(
                                        height=250
                                    )
                                    
                                    st.altair_chart(chart, use_container_width=True)
                            
                            with col2:
                                # Statistics summary
                                st.metric("Average", f"{stat_data['Value'].mean():,.1f}")
                                st.metric("Min", f"{stat_data['Value'].min():,.1f}")
                                st.metric("Max", f"{stat_data['Value'].max():,.1f}")
                                st.metric("Std Dev", f"{stat_data['Value'].std():,.1f}")
                            
                            st.markdown("---")
                
                else:
                    # Single statistic - show distribution
                    st.markdown(f"#### {st.session_state.selected_statistics[0]} Distribution")
                    
                    if len(data['year'].unique()) >= 3:
                        # Box plot across years
                        chart = alt.Chart(data).mark_boxplot(extent='min-max').encode(
                            x=alt.X('year:O', title='Year'),
                            y=alt.Y('Value:Q', title='Value'),
                            color=alt.value('#2E7D32')
                        ).properties(
                            height=400,
                            title='Distribution by Year'
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        st.info("Box plot shows median (center line), quartiles (box), and min/max (whiskers)")
            
            with tab4:
                st.markdown("### Data Table & Export")
                
                # Help expander explaining columns
                with st.expander("Understanding Your Data Columns", expanded=False):
                    st.markdown("""
                    **What do these columns mean?**
                    
                    - **Value**: The actual measurement (expense, production, yield, price, etc.)
                    - **Data Reliability (%)**: Coefficient of Variation - measures data quality
                      - **<15%** = Very reliable data ✅
                      - **15-25%** = Good quality data ⚠️
                      - **>25%** = Use with caution ❗
                    - **Unit**: Measurement unit (dollars, acres, bushels, pounds, etc.)
                    - **Year**: Data collection year
                    - **State/County/Region**: Geographic location
                    
                    Tip: We automatically show only the most relevant columns for your selection.
                    """)
                
                # Clean data display
                st.markdown("#### Clean Data View (Recommended)")
                st.caption("Only essential columns with user-friendly names")
                
                clean_data = clean_dataframe_for_display(data, display_columns, friendly_names)
                # Column selector
                selected_columns = st.multiselect(
                    "Columns to display",
                    options=list(clean_data.columns),
                    default=list(clean_data.columns),
                    key="clean_columns_selector"
                )
                filtered_data = clean_data[selected_columns] if selected_columns else clean_data
                # Pagination controls
                col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
                with col_p1:
                    page_size = st.selectbox("Rows per page", options=[25, 50, 100], index=0, key="page_size")
                total_rows = len(filtered_data)
                total_pages = max(1, (total_rows + page_size - 1) // page_size)
                with col_p2:
                    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="page_number")
                start = (page_num - 1) * page_size
                end = start + page_size
                st.dataframe(filtered_data.iloc[start:end], height=400, use_container_width=True)
                
                st.markdown("---")
                
                # Export options
                st.markdown("#### Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Clean CSV export
                    clean_csv = clean_data.to_csv(index=False)
                    st.download_button(
                        label="Download Clean Data",
                        data=clean_csv,
                        file_name=f"usda_{st.session_state.selected_commodity.lower().replace(' ', '_')}_clean.csv",
                        mime="text/csv",
                        help="User-friendly column names, only essential data",
                        type="primary"
                    )
                
                with col2:
                    # Full CSV export
                    full_csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download All Columns",
                        data=full_csv,
                        file_name=f"usda_{st.session_state.selected_commodity.lower().replace(' ', '_')}_full.csv",
                        mime="text/csv",
                        help="All original API columns for advanced analysis"
                    )
                
                with col3:
                    # Summary statistics export
                    summary = data.groupby(['year'])['Value'].agg(['mean', 'min', 'max', 'std']).reset_index()
                    summary.columns = ['Year', 'Average', 'Minimum', 'Maximum', 'Std Deviation']
                    summary_csv = summary.to_csv(index=False)
                    st.download_button(
                        label="Download Summary",
                        data=summary_csv,
                        file_name=f"usda_{st.session_state.selected_commodity.lower().replace(' ', '_')}_summary.csv",
                        mime="text/csv",
                        help="Statistical summary by year"
                    )
                
                # Advanced: Show all columns toggle
                st.markdown("---")
                if st.checkbox("Show All Raw Columns (Advanced)", value=False):
                    st.markdown("#### Complete Raw Data")
                    st.caption(f"All {len(data.columns)} columns from USDA API")
                    st.dataframe(data, height=400, use_container_width=True)
        
        else:
            # Instructions when no data loaded
            st.markdown("## Welcome to the USDA Data Explorer")
            st.markdown("---")
            
            st.markdown("""
            ### How to Use This Dashboard
            
            This dynamic dashboard guides you through data exploration step-by-step:
            
            1. **📦 Select Category** - Choose from multiple sectors:
               - 🌽 Field Crops, 🍎 Fruits, 🥕 Vegetables, 🥜 Nuts & Seeds
               - 🐄 Animals, 🥛 Animal Products, 🌿 Specialty Crops
               - 💰 **Economics & Expenses** (Fertilizer, Labor, Fuel costs)
               - 🌍 **Environmental Data** (Chemical applications, Conservation)
               - 📊 **Farm Operations** (Farm counts, Operator demographics)
            2. **🌽 Select Commodity** - Pick a specific commodity from your category
            3. **📊 Select Statistics** - Choose what to measure (Price, Yield, Production, Expenses, etc.)
            4. **📍 Select Geographic Level** - Choose National, State, or County level
            5. **🗺️ Select Locations** - Pick specific states or counties to analyze
            6. **📅 Select Years** - Choose specific years or use quick-select buttons
            
            ### ✨ Smart Features
            
            - ✅ **Dynamic Filtering**: Only see options that have actual data
            - ✅ **Data Availability**: See record counts and year ranges
            - ✅ **Multi-Select**: Compare multiple locations and statistics
            - ✅ **Progress Tracking**: Know exactly where you are in the process
            - ✅ **No Dead Ends**: Can't select invalid combinations
            
            ### 🚀 Get Started
            
            Begin by selecting a commodity category in the sidebar! ➡️
            """)
            
            # Show some stats about available data
            st.markdown("---")
            st.markdown("### Available Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**{len(all_commodities)}**\n\nCommodities Available")
            with col2:
                st.info(f"**{len(categorized)}**\n\nCategories")
            with col3:
                st.info("**1990-2024**\n\nHistorical Data")


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
