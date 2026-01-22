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
        """Load and process facility inventory data with hierarchical structure"""
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
        
        print(f"  Loaded {len(facility_df)} facilities and {len(systems_df)} systems")
        
        return facility_df, systems_df
    
    def group_systems_by_facility(self, systems_df, facility_df):
        """
        Group systems by their parent facility
        
        Args:
            systems_df: System condition dataframe
            facility_df: Facility condition dataframe
            
        Returns:
            Dictionary mapping facility_id to list of systems
        """
        print("Grouping systems by facility...")
        
        # Identify facility ID column in systems
        facility_id_col = None
        for col in ['Facility Number', 'Facility ID', 'Parent Facility', 'Facility', 'Unique Identifier']:
            if col in systems_df.columns:
                facility_id_col = col
                break
        
        # If no explicit facility column, try to infer from system ID pattern
        if facility_id_col is None:
            # Check if system IDs follow pattern that can be mapped to facilities
            system_id_col = None
            for col in ['Unique Identifier', 'System ID', 'ID']:
                if col in systems_df.columns:
                    system_id_col = col
                    break
            
            if system_id_col:
                # Try to extract facility number from system ID (e.g., SYS-001 -> Facility #1)
                # This is a heuristic - adjust based on actual data structure
                systems_df['_inferred_facility'] = systems_df[system_id_col].apply(
                    lambda x: self._extract_facility_from_system_id(x)
                )
                facility_id_col = '_inferred_facility'
        
        # If still no facility mapping, create default grouping
        if facility_id_col is None:
            print("  WARNING: Could not identify facility grouping. Using default mapping.")
            # Group systems sequentially (e.g., 15 systems per facility)
            systems_per_facility = 15
            systems_df['_grouped_facility'] = (systems_df.index // systems_per_facility).apply(
                lambda x: f"FAC-{x+1:03d}"
            )
            facility_id_col = '_grouped_facility'
        
        # Group systems by facility
        facility_systems = {}
        for facility_id, group in systems_df.groupby(facility_id_col):
            facility_systems[facility_id] = group.to_dict('records')
        
        print(f"  Grouped {len(systems_df)} systems into {len(facility_systems)} facilities")
        
        return facility_systems, facility_id_col
    
    def _extract_facility_from_system_id(self, system_id):
        """Extract facility number from system ID (heuristic)"""
        if pd.isna(system_id):
            return None
        
        system_id_str = str(system_id)
        # Try patterns like SYS-001 -> Facility #1, or Facility1-SYS-001 -> Facility1
        match = re.search(r'Facility\s*#?(\d+)', system_id_str, re.IGNORECASE)
        if match:
            return f"FAC-{int(match.group(1)):03d}"
        
        # Try to extract number and map to facility
        match = re.search(r'(\d+)', system_id_str)
        if match:
            # Assume systems 1-15 belong to Facility 1, 16-30 to Facility 2, etc.
            sys_num = int(match.group(1))
            facility_num = ((sys_num - 1) // 15) + 1
            return f"FAC-{facility_num:03d}"
        
        return None
    
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
        Merge all datasets into master training data with hierarchical structure
        
        Creates both facility-level and system-level rows with 'level' column
        
        Args:
            facility_df: Facility inventory
            systems_df: System condition data
            wo_df: Processed work orders
            synthetic_df: Synthetic failures
            
        Returns:
            Merged master dataframe with 'level' column (facility/system)
        """
        print("Merging datasets with hierarchical structure...")
        
        # Combine real and synthetic failures
        all_failures = pd.concat([
            wo_df[['facility_id', 'occurred_date', 'failure_year', 'failure_type', 'component']].assign(is_synthetic=False),
            synthetic_df
        ], ignore_index=True)
        
        # Group systems by facility
        facility_systems_dict, facility_id_col = self.group_systems_by_facility(systems_df, facility_df)
        
        # Standardize facility ID in facility_df
        if 'Unique Identifier' in facility_df.columns:
            facility_df = facility_df.rename(columns={'Unique Identifier': 'facility_id'})
        elif 'facility_id' not in facility_df.columns:
            # Try to find facility ID column
            for col in ['Facility Number', 'Facility ID', 'ID']:
                if col in facility_df.columns:
                    facility_df = facility_df.rename(columns={col: 'facility_id'})
                    break
        
        # Prepare systems_df with facility_id
        systems_processed = systems_df.copy()
        if facility_id_col and facility_id_col in systems_processed.columns:
            systems_processed['facility_id'] = systems_processed[facility_id_col]
        elif '_inferred_facility' in systems_processed.columns:
            systems_processed['facility_id'] = systems_processed['_inferred_facility']
        elif '_grouped_facility' in systems_processed.columns:
            systems_processed['facility_id'] = systems_processed['_grouped_facility']
        else:
            # Default: assign systems to facilities sequentially
            num_facilities = len(facility_df)
            systems_per_fac = len(systems_processed) // max(num_facilities, 1)
            facility_ids = []
            for i in range(len(systems_processed)):
                fac_idx = min(i // max(systems_per_fac, 1), num_facilities - 1)
                if 'facility_id' in facility_df.columns:
                    facility_ids.append(facility_df.iloc[fac_idx]['facility_id'])
                else:
                    facility_ids.append(f"FAC-{fac_idx+1:03d}")
            systems_processed['facility_id'] = facility_ids
        
        # Get system ID column
        system_id_col = None
        for col in ['Unique Identifier', 'System ID', 'ID']:
            if col in systems_processed.columns:
                system_id_col = col
                break
        
        if system_id_col:
            systems_processed = systems_processed.rename(columns={system_id_col: 'entity_id'})
        else:
            systems_processed['entity_id'] = systems_processed.index.map(lambda x: f"SYS-{x+1:03d}")
        
        # Aggregate failures by facility and by system
        # Facility-level failures
        facility_failures = all_failures.groupby('facility_id').agg({
            'occurred_date': ['count', 'min', 'max'],
            'failure_type': lambda x: x.mode()[0] if len(x) > 0 else 'none'
        }).reset_index()
        facility_failures.columns = ['facility_id', 'total_failures', 'first_failure_date', 
                                     'last_failure_date', 'most_common_failure']
        
        # System-level failures (if we have system IDs in failures)
        system_failures = pd.DataFrame()
        if 'system_id' in all_failures.columns or 'entity_id' in all_failures.columns:
            sys_id_col = 'system_id' if 'system_id' in all_failures.columns else 'entity_id'
            system_failures = all_failures.groupby(sys_id_col).agg({
                'occurred_date': ['count', 'min', 'max'],
                'failure_type': lambda x: x.mode()[0] if len(x) > 0 else 'none'
            }).reset_index()
            system_failures.columns = [sys_id_col, 'total_failures', 'first_failure_date',
                                      'last_failure_date', 'most_common_failure']
        
        # Create facility-level rows
        facility_rows = []
        for _, facility in facility_df.iterrows():
            fac_id = facility.get('facility_id', facility.get('Unique Identifier', f"FAC-{len(facility_rows)+1:03d}"))
            
            # Get systems for this facility
            facility_systems = systems_processed[systems_processed['facility_id'] == fac_id]
            
            # Aggregate facility metrics from systems
            if len(facility_systems) > 0:
                # Weighted average condition (by life expectancy)
                if 'Condition Index' in facility_systems.columns and 'Life Expectancy' in facility_systems.columns:
                    weights = facility_systems['Life Expectancy'].fillna(50)
                    facility_condition = (facility_systems['Condition Index'] * weights).sum() / weights.sum()
                elif 'Condition Index' in facility_systems.columns:
                    facility_condition = facility_systems['Condition Index'].mean()
                else:
                    facility_condition = facility.get('Condition Index', 65)
                
                facility_age = facility_systems['Age (years)'].max() if 'Age (years)' in facility_systems.columns else facility.get('Age (years)', 20)
                facility_life_expectancy = facility_systems['Life Expectancy'].mean() if 'Life Expectancy' in facility_systems.columns else facility.get('Life Expectancy', 50)
            else:
                facility_condition = facility.get('Condition Index', 65)
                facility_age = facility.get('Age (years)', 20)
                facility_life_expectancy = facility.get('Life Expectancy', 50)
            
            # Merge facility failures
            fac_failures = facility_failures[facility_failures['facility_id'] == fac_id]
            if len(fac_failures) > 0:
                total_failures = int(fac_failures.iloc[0]['total_failures'])
                first_failure = fac_failures.iloc[0]['first_failure_date']
                last_failure = fac_failures.iloc[0]['last_failure_date']
                most_common = fac_failures.iloc[0]['most_common_failure']
            else:
                total_failures = 0
                first_failure = pd.NaT
                last_failure = pd.NaT
                most_common = 'none'
            
            days_between = ((last_failure - first_failure).days / total_failures) if total_failures > 0 and pd.notna(last_failure) and pd.notna(first_failure) else 365
            
            facility_row = {
                'level': 'facility',
                'parent_id': None,
                'entity_id': fac_id,
                'facility_id': fac_id,
                'Type': facility.get('Type', 'Facility'),
                'Title': facility.get('Title', facility.get('Name', fac_id)),
                'Condition Index': round(facility_condition, 1),
                'Age (years)': facility_age,
                'Life Expectancy': facility_life_expectancy,
                'total_failures': total_failures,
                'first_failure_date': first_failure,
                'last_failure_date': last_failure,
                'most_common_failure': most_common,
                'days_between_failures': days_between
            }
            
            # Add other facility columns
            for col in facility.index:
                if col not in facility_row:
                    facility_row[col] = facility[col]
            
            facility_rows.append(facility_row)
        
        # Create system-level rows
        system_rows = []
        for _, system in systems_processed.iterrows():
            sys_id = system.get('entity_id', f"SYS-{len(system_rows)+1:03d}")
            fac_id = system.get('facility_id', None)
            
            # Get system failures
            sys_failures = system_failures[system_failures.get('entity_id', system_failures.columns[0]) == sys_id] if len(system_failures) > 0 else pd.DataFrame()
            if len(sys_failures) > 0:
                total_failures = int(sys_failures.iloc[0]['total_failures'])
                first_failure = sys_failures.iloc[0]['first_failure_date']
                last_failure = sys_failures.iloc[0]['last_failure_date']
                most_common = sys_failures.iloc[0]['most_common_failure']
            else:
                # Use facility-level failures as fallback
                if fac_id:
                    fac_failures = facility_failures[facility_failures['facility_id'] == fac_id]
                    if len(fac_failures) > 0:
                        total_failures = int(fac_failures.iloc[0]['total_failures'] // len(systems_processed[systems_processed['facility_id'] == fac_id]))
                    else:
                        total_failures = 0
                else:
                    total_failures = 0
                first_failure = pd.NaT
                last_failure = pd.NaT
                most_common = 'none'
            
            days_between = ((last_failure - first_failure).days / total_failures) if total_failures > 0 and pd.notna(last_failure) and pd.notna(first_failure) else 365
            
            system_row = {
                'level': 'system',
                'parent_id': fac_id,
                'entity_id': sys_id,
                'facility_id': fac_id,
                'Type': system.get('Type', system.get('System Type', 'System')),
                'Title': system.get('Title', system.get('Name', sys_id)),
                'Condition Index': system.get('Condition Index', 65),
                'Age (years)': system.get('Age (years)', 20),
                'Life Expectancy': system.get('Life Expectancy', 50),
                'total_failures': total_failures,
                'first_failure_date': first_failure,
                'last_failure_date': last_failure,
                'most_common_failure': most_common,
                'days_between_failures': days_between
            }
            
            # Add other system columns
            for col in system.index:
                if col not in system_row and col not in ['_inferred_facility', '_grouped_facility']:
                    system_row[col] = system[col]
            
            system_rows.append(system_row)
        
        # Combine facility and system rows
        master = pd.DataFrame(facility_rows + system_rows)
        
        # Calculate derived features for all rows
        master['failure_rate'] = master['total_failures'] / master['Age (years)'].clip(lower=1)
        master['condition_delta'] = 100 - master['Condition Index']
        
        age_col = master['Age (years)']
        life_expectancy_col = master['Life Expectancy']
        master['risk_score'] = (
            master['failure_rate'] * 0.3 +
            master['condition_delta'] * 0.4 +
            (age_col / life_expectancy_col) * 100 * 0.3
        )
        
        # Target variable: Probability of failure within 12 months
        master['failure_within_12mo'] = (
            (master['total_failures'] > 0) & 
            (pd.notna(master['last_failure_date'])) &
            ((datetime.now() - pd.to_datetime(master['last_failure_date'])).dt.days < 365)
        ).astype(int)
        
        print(f"  Created {len(facility_rows)} facility rows and {len(system_rows)} system rows")
        
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
