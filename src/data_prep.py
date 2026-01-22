"""
Phase 1: Data Preparation ETL Script
Merges facility inventory with failure history and extracts hidden patterns

This script performs:
1. Regex-based date extraction from work order text
2. Failure type classification
3. Synthetic failure generation for facilities without history
4. Feature engineering and data merging
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FacilityDataETL:
    def __init__(self, inventory_path, workorder_path):
        """
        Initialize ETL processor
        
        Args:
            inventory_path: Path to generated_facility_data.xlsx
            workorder_path: Path to Simulated_Data.xlsx
        """
        self.inventory_path = inventory_path
        self.workorder_path = workorder_path
        self.master_data = None
        
    def load_inventory(self):
        """Load and process facility inventory data"""
        print("Loading facility inventory...")
        
        # Load Facility Condition Index sheet
        facility_df = pd.read_excel(
            self.inventory_path, 
            sheet_name='Facility Condition Index'
        )
        
        # Load System Condition Index sheet
        systems_df = pd.read_excel(
            self.inventory_path,
            sheet_name='System Condition Index'
        )
        
        return facility_df, systems_df
    
    def load_work_orders(self):
        """Load work order failure history"""
        print("Loading work order history...")
        
        wo_df = pd.read_excel(
            self.workorder_path,
            sheet_name='Infra Work Orders'
        )
        
        return wo_df
    
    def extract_failure_dates(self, text):
        """
        Extract dates from work order scope text using regex patterns
        
        Supports multiple date formats:
        - "Occurred: Feb 2025" → 2025-02-01
        - "Occurred: February 15, 2025" → 2025-02-15
        - "2025-02-15" (ISO format) → 2025-02-15
        
        Args:
            text: Scope description text
            
        Returns:
            dict with extracted date info or None
        """
        if pd.isna(text):
            return None
        
        text = str(text)
        
        # Pattern 1: "Occurred: Month Year" or "Occurred Month Day, Year"
        occurred_pattern = r'Occurred[:\s]+([A-Za-z]+\s+\d{1,2}[,\s]+\d{4}|[A-Za-z]+\s+\d{4})'
        
        # Pattern 2: "Month Day, Year" or "Month Year"
        date_pattern = r'\b([A-Za-z]+)\s+(\d{1,2})[,\s]+(\d{4})\b|\b([A-Za-z]+)\s+(\d{4})\b'
        
        # Pattern 3: ISO format YYYY-MM-DD
        iso_pattern = r'\b(\d{4})-(\d{2})-(\d{2})\b'
        
        result = {
            'occurred_date': None,
            'year': None,
            'month': None,
            'day': None
        }
        
        # Try occurred pattern first
        match = re.search(occurred_pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                # Try parsing with day
                parsed = pd.to_datetime(date_str, format='%B %d, %Y', errors='coerce')
                if pd.isna(parsed):
                    # Try without day
                    parsed = pd.to_datetime(date_str, format='%B %Y', errors='coerce')
                
                if not pd.isna(parsed):
                    result['occurred_date'] = parsed
                    result['year'] = parsed.year
                    result['month'] = parsed.month
                    result['day'] = parsed.day if parsed.day != 1 else None
                    return result
            except:
                pass
        
        # Try general date patterns
        matches = re.finditer(date_pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if match.group(1):  # Month Day, Year format
                    month_str, day_str, year_str = match.group(1, 2, 3)
                    parsed = pd.to_datetime(f"{month_str} {day_str}, {year_str}", errors='coerce')
                else:  # Month Year format
                    month_str, year_str = match.group(4, 5)
                    parsed = pd.to_datetime(f"{month_str} {year_str}", errors='coerce')
                
                if not pd.isna(parsed):
                    result['occurred_date'] = parsed
                    result['year'] = parsed.year
                    result['month'] = parsed.month
                    result['day'] = parsed.day if match.group(2) else None
                    return result
            except:
                continue
        
        # Try ISO format
        match = re.search(iso_pattern, text)
        if match:
            year, month, day = match.groups()
            try:
                parsed = pd.to_datetime(f"{year}-{month}-{day}")
                result['occurred_date'] = parsed
                result['year'] = int(year)
                result['month'] = int(month)
                result['day'] = int(day)
                return result
            except:
                pass
        
        return result if result['occurred_date'] else None
    
    def extract_failure_type(self, text):
        """
        Extract failure type/cause from scope text
        
        Failure types detected:
        - compressor, electrical, leak, HVAC, structural
        - plumbing, roofing, fire_protection, wear, age_related
        
        Args:
            text: Scope description text
            
        Returns:
            dict with failure classification
        """
        if pd.isna(text):
            return {'failure_type': 'Unknown', 'component': 'Unknown'}
        
        text = str(text).lower()
        
        # Define failure patterns with regex
        failure_patterns = {
            'compressor': r'compressor\s+(failure|failed|broke)',
            'electrical': r'(electrical|electric|power)\s+(failure|outage|short)',
            'leak': r'(leak|leaking|water damage|flooding)',
            'hvac': r'(hvac|heating|cooling|air\s+conditioning)\s+(failure|issue|problem)',
            'structural': r'(crack|structural|foundation|settling)',
            'plumbing': r'(plumbing|pipe|drain)\s+(failure|break|clog)',
            'roofing': r'(roof|roofing)\s+(leak|damage|failure)',
            'fire_protection': r'(fire\s+protection|sprinkler|alarm)\s+(failure|issue)',
            'wear': r'(wear|deterioration|degradation)',
            'age_related': r'(age|aging|old|obsolete)'
        }
        
        for failure_type, pattern in failure_patterns.items():
            if re.search(pattern, text):
                # Extract component if possible
                component_match = re.search(r'(compressor|transformer|pump|valve|panel|unit|system)', text)
                component = component_match.group(1) if component_match else failure_type
                
                return {
                    'failure_type': failure_type,
                    'component': component
                }
        
        return {'failure_type': 'general', 'component': 'unknown'}
    
    def process_work_orders(self, wo_df):
        """
        Process work orders to extract failure patterns
        
        Args:
            wo_df: Work order dataframe
            
        Returns:
            Processed dataframe with extracted features
        """
        print("Extracting failure patterns from work orders...")
        
        # Extract dates and failure types
        wo_df['date_info'] = wo_df['Scope'].apply(self.extract_failure_dates)
        wo_df['failure_info'] = wo_df['Scope'].apply(self.extract_failure_type)
        
        # Expand nested dictionaries
        date_expanded = pd.json_normalize(wo_df['date_info'].dropna())
        failure_expanded = pd.json_normalize(wo_df['failure_info'])
        
        # Merge back
        wo_df = pd.concat([
            wo_df.reset_index(drop=True),
            date_expanded.reset_index(drop=True),
            failure_expanded.reset_index(drop=True)
        ], axis=1)
        
        # Clean up
        wo_df['failure_year'] = pd.to_datetime(wo_df['occurred_date']).dt.year
        
        # Map facility numbers (handling various ID formats)
        if 'Facility Number' in wo_df.columns:
            wo_df['facility_id'] = wo_df['Facility Number']
        elif 'Unique Identifier' in wo_df.columns:
            wo_df['facility_id'] = wo_df['Unique Identifier']
        else:
            # Extract from scope text
            wo_df['facility_id'] = wo_df['Scope'].apply(
                lambda x: re.search(r'Facility\s+(\d+|[A-Z]+-\d+)', str(x), re.IGNORECASE).group(1) 
                if re.search(r'Facility\s+(\d+|[A-Z]+-\d+)', str(x), re.IGNORECASE) else None
            )
        
        return wo_df
    
    def calculate_failure_statistics(self, wo_df):
        """
        Calculate failure rate statistics by facility and system type
        
        Args:
            wo_df: Processed work order dataframe
            
        Returns:
            Statistical summary dataframe
        """
        print("Calculating failure statistics...")
        
        stats = {}
        
        # Overall failure rate
        if 'failure_year' in wo_df.columns and wo_df['failure_year'].notna().any():
            year_range = wo_df['failure_year'].max() - wo_df['failure_year'].min() + 1
            total_failures = len(wo_df)
            stats['avg_failures_per_year'] = total_failures / max(year_range, 1)
        
        # Failure by type
        if 'failure_type' in wo_df.columns:
            type_counts = wo_df['failure_type'].value_counts()
            stats['failure_type_distribution'] = type_counts.to_dict()
        
        # Failure by facility
        if 'facility_id' in wo_df.columns and wo_df['facility_id'].notna().any():
            facility_counts = wo_df.groupby('facility_id').size()
            if len(facility_counts) > 0:
                stats['avg_failures_per_facility'] = float(facility_counts.mean())
                stats['max_failures_per_facility'] = int(facility_counts.max())
            else:
                stats['avg_failures_per_facility'] = 2.0
                stats['max_failures_per_facility'] = 5
        else:
            stats['avg_failures_per_facility'] = 2.0
            stats['max_failures_per_facility'] = 5
        
        return stats
    
    def generate_synthetic_failures(self, facility_df, wo_stats, num_failures=None):
        """
        Generate synthetic failure history for facilities without recorded failures
        
        Uses statistical distributions based on:
        - Facility age
        - Condition index
        - Historical failure patterns
        
        Args:
            facility_df: Facility inventory dataframe
            wo_stats: Work order statistics
            num_failures: Number of synthetic failures to generate
            
        Returns:
            Synthetic failure dataframe
        """
        print("Generating synthetic failure history...")
        
        if num_failures is None:
            # Get avg_failures_per_facility, handling NaN values
            avg_failures = wo_stats.get('avg_failures_per_facility', 2.0)
            # Check if it's NaN or None
            if avg_failures is None or (isinstance(avg_failures, float) and np.isnan(avg_failures)):
                avg_failures = 2.0
            num_failures = int(len(facility_df) * avg_failures)
        
        synthetic_failures = []
        
        # Get failure type distribution
        failure_types = wo_stats.get('failure_type_distribution', {})
        if not failure_types:
            failure_types = {'general': 1}
        
        total_weight = sum(failure_types.values())
        failure_probs = {k: v/total_weight for k, v in failure_types.items()}
        
        for idx, facility in facility_df.iterrows():
            # Determine number of failures based on facility age and condition
            if 'Age (years)' in facility and 'Condition Index' in facility:
                age = facility.get('Age (years)', 10)
                condition = facility.get('Condition Index', 65)
                
                # More failures for older, lower-condition facilities
                expected_failures = max(1, int(age / 10) * (100 - condition) / 20)
            else:
                expected_failures = np.random.poisson(2)
            
            for _ in range(expected_failures):
                # Random failure type based on distribution
                failure_type = np.random.choice(
                    list(failure_probs.keys()),
                    p=list(failure_probs.values())
                )
                
                # Random date within last 5 years
                days_back = np.random.randint(0, 1825)  # 5 years
                failure_date = datetime.now() - timedelta(days=days_back)
                
                synthetic_failures.append({
                    'facility_id': facility.get('Unique Identifier', f'FAC-{idx}'),
                    'occurred_date': failure_date,
                    'failure_year': failure_date.year,
                    'failure_type': failure_type,
                    'component': 'synthetic',
                    'is_synthetic': True
                })
        
        return pd.DataFrame(synthetic_failures)
    
    def merge_inventory_and_failures(self, facility_df, systems_df, wo_df, synthetic_df):
        """
        Merge all datasets into master training data
        
        Args:
            facility_df: Facility inventory
            systems_df: System condition data
            wo_df: Processed work orders
            synthetic_df: Synthetic failures
            
        Returns:
            Merged master dataframe
        """
        print("Merging datasets...")
        
        # Combine real and synthetic failures
        all_failures = pd.concat([
            wo_df[['facility_id', 'occurred_date', 'failure_year', 'failure_type', 'component']].assign(is_synthetic=False),
            synthetic_df
        ], ignore_index=True)
        
        # Aggregate failures by facility
        failure_summary = all_failures.groupby('facility_id').agg({
            'occurred_date': ['count', 'min', 'max'],
            'failure_type': lambda x: x.mode()[0] if len(x) > 0 else 'none'
        }).reset_index()
        
        failure_summary.columns = ['facility_id', 'total_failures', 'first_failure_date', 
                                   'last_failure_date', 'most_common_failure']
        
        # Calculate time between failures
        failure_summary['days_between_failures'] = (
            (failure_summary['last_failure_date'] - failure_summary['first_failure_date']).dt.days / 
            failure_summary['total_failures'].clip(lower=1)
        )
        
        # Merge with facility inventory
        master = facility_df.copy()
        
        # Standardize facility ID column
        if 'Unique Identifier' in master.columns:
            master.rename(columns={'Unique Identifier': 'facility_id'}, inplace=True)
        
        # Merge failure data
        master = master.merge(failure_summary, on='facility_id', how='left')
        
        # Fill missing values
        master['total_failures'] = master['total_failures'].fillna(0)
        master['most_common_failure'] = master['most_common_failure'].fillna('none')
        
        # Add system-level features
        if 'facility_id' in systems_df.columns or 'Unique Identifier' in systems_df.columns:
            systems_agg = systems_df.groupby(
                systems_df.columns[0]  # First column is usually the ID
            ).agg({
                'Condition Index': ['mean', 'min', 'std'],
                'Life Expectancy': 'mean'
            }).reset_index()
            
            systems_agg.columns = ['facility_id', 'avg_system_condition', 'min_system_condition',
                                   'system_condition_std', 'avg_system_life_expectancy']
            
            master = master.merge(systems_agg, on='facility_id', how='left')
        
        # Calculate derived features
        # Handle failure_rate calculation
        if 'Age (years)' in master.columns:
            master['failure_rate'] = master['total_failures'] / master['Age (years)'].clip(lower=1)
        else:
            master['failure_rate'] = master['total_failures'] / 1
        
        # Handle condition_delta
        if 'Condition Index' in master.columns:
            master['condition_delta'] = 100 - master['Condition Index']
        else:
            master['condition_delta'] = 35  # Default if condition index missing
        
        # Handle risk_score calculation
        age_col = master.get('Age (years)', pd.Series([20] * len(master))) if 'Age (years)' not in master.columns else master['Age (years)']
        life_expectancy_col = master.get('Life Expectancy', pd.Series([50] * len(master))) if 'Life Expectancy' not in master.columns else master['Life Expectancy']
        
        master['risk_score'] = (
            master['failure_rate'] * 0.3 +
            master['condition_delta'] * 0.4 +
            (age_col / life_expectancy_col) * 100 * 0.3
        )
        
        # Target variable: Probability of failure within 12 months
        master['failure_within_12mo'] = (
            (master['total_failures'] > 0) & 
            ((datetime.now() - master['last_failure_date']).dt.days < 365)
        ).astype(int)
        
        return master
    
    def save_master_data(self, output_path='data/Master_Training_Data.csv'):
        """Save processed master dataset"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.master_data.to_csv(output_file, index=False)
        print(f"\n✓ Master training data saved to: {output_file}")
        print(f"  Total records: {len(self.master_data)}")
        print(f"  Total features: {len(self.master_data.columns)}")
        
        return output_file
    
    def run_etl(self):
        """Execute complete ETL pipeline"""
        print("="*60)
        print("PHASE 1: DATA PREPARATION ETL")
        print("="*60 + "\n")
        
        # Load data
        facility_df, systems_df = self.load_inventory()
        wo_df = self.load_work_orders()
        
        # Process work orders
        wo_processed = self.process_work_orders(wo_df)
        
        # Calculate statistics
        wo_stats = self.calculate_failure_statistics(wo_processed)
        
        print("\nFailure Statistics:")
        for key, value in wo_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in list(value.items())[:5]:
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value:.2f}")
        
        # Generate synthetic failures
        synthetic_df = self.generate_synthetic_failures(facility_df, wo_stats)
        
        # Merge all data
        self.master_data = self.merge_inventory_and_failures(
            facility_df, systems_df, wo_processed, synthetic_df
        )
        
        # Save output
        output_path = self.save_master_data()
        
        print("\n" + "="*60)
        print("ETL COMPLETE")
        print("="*60)
        
        return self.master_data, output_path


# Main execution
if __name__ == "__main__":
    # File paths
    INVENTORY_FILE = "data/generated_facility_data.xlsx"
    WORKORDER_FILE = "data/Simulated_Data.xlsx"
    
    # Check if files exist
    if not Path(INVENTORY_FILE).exists():
        print(f"ERROR: {INVENTORY_FILE} not found!")
        print("Please ensure the inventory file exists in the data/ directory.")
        exit(1)
    
    if not Path(WORKORDER_FILE).exists():
        print(f"ERROR: {WORKORDER_FILE} not found!")
        print("Please ensure the work order file exists in the data/ directory.")
        exit(1)
    
    # Run ETL
    etl = FacilityDataETL(INVENTORY_FILE, WORKORDER_FILE)
    master_data, output_path = etl.run_etl()
    
    # Display sample
    print("\nSample of master data:")
    print(master_data.head())
    print("\nColumns:", list(master_data.columns))
