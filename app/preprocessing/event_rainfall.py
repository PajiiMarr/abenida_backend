# backend/app/preprocessing/event_rainfall.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EventRainfallExtractor:
    """Extract rainfall for specific flood events from daily data"""
    
    def __init__(self, chirps_daily_path: Path, gpm_daily_path: Path):
        self.chirps = pd.read_csv(chirps_daily_path)
        self.gpm = pd.read_csv(gpm_daily_path)
        
        # Parse dates
        self.chirps['date'] = pd.to_datetime(self.chirps['date'])
        self.gpm['date'] = pd.to_datetime(self.gpm['date'])
    
    def extract_event_features(self, event_config: dict) -> dict:
        """
        Extract rainfall features for a single event from config
        
        Args:
            event_config: Dictionary with 'date', 'days_before', 'days_after'
            
        Returns:
            Dictionary of rainfall features
        """
        event_date = event_config['date']
        days_before = event_config.get('days_before', 7)
        days_after = event_config.get('days_after', 7)
        event_key = event_config.get('sar_key', '')
        
        start_date = event_date - timedelta(days=days_before)
        end_date = event_date + timedelta(days=days_after)
        
        # Filter CHIRPS
        chirps_event = self.chirps[
            (self.chirps['date'] >= start_date) & 
            (self.chirps['date'] <= end_date)
        ]
        
        # Filter GPM
        gpm_event = self.gpm[
            (self.gpm['date'] >= start_date) & 
            (self.gpm['date'] <= end_date)
        ]
        
        # Calculate pre-event (7 days before)
        chirps_pre = self.chirps[
            (self.chirps['date'] >= event_date - timedelta(days=days_before)) & 
            (self.chirps['date'] < event_date)
        ]
        gpm_pre = self.gpm[
            (self.gpm['date'] >= event_date - timedelta(days=days_before)) & 
            (self.gpm['date'] < event_date)
        ]
        
        # Calculate post-event (7 days after)
        chirps_post = self.chirps[
            (self.chirps['date'] > event_date) & 
            (self.chirps['date'] <= event_date + timedelta(days=days_after))
        ]
        gpm_post = self.gpm[
            (self.gpm['date'] > event_date) & 
            (self.gpm['date'] <= event_date + timedelta(days=days_after))
        ]
        
        # Build features with dynamic naming
        prefix = event_key if event_key else f"event_{event_date.strftime('%Y%m%d')}"
        
        features = {
            f'{prefix}_chirps_mean': chirps_event['rainfall_mm'].mean() if len(chirps_event) > 0 else 0,
            f'{prefix}_chirps_max': chirps_event['rainfall_mm'].max() if len(chirps_event) > 0 else 0,
            f'{prefix}_chirps_sum': chirps_event['rainfall_mm'].sum() if len(chirps_event) > 0 else 0,
            f'{prefix}_chirps_std': chirps_event['rainfall_mm'].std() if len(chirps_event) > 0 else 0,
            f'{prefix}_gpm_mean': gpm_event['rainfall_mm'].mean() if len(gpm_event) > 0 else 0,
            f'{prefix}_gpm_max': gpm_event['rainfall_mm'].max() if len(gpm_event) > 0 else 0,
            f'{prefix}_gpm_sum': gpm_event['rainfall_mm'].sum() if len(gpm_event) > 0 else 0,
            f'{prefix}_gpm_std': gpm_event['rainfall_mm'].std() if len(gpm_event) > 0 else 0,
            f'{prefix}_pre_rain': chirps_pre['rainfall_mm'].sum() if len(chirps_pre) > 0 else 0,
            f'{prefix}_post_rain': chirps_post['rainfall_mm'].sum() if len(chirps_post) > 0 else 0,
            f'{prefix}_gpm_pre_rain': gpm_pre['rainfall_mm'].sum() if len(gpm_pre) > 0 else 0,
            f'{prefix}_gpm_post_rain': gpm_post['rainfall_mm'].sum() if len(gpm_post) > 0 else 0,
            f'{prefix}_total_rain': chirps_event['rainfall_mm'].sum() if len(chirps_event) > 0 else 0,
            f'{prefix}_rain_days': len(chirps_event[chirps_event['rainfall_mm'] > 1]) if len(chirps_event) > 0 else 0,
        }
        
        return features


def get_event_rainfall_features(event_config: dict) -> dict:
    """
    Get rainfall features for a single event from config
    
    Args:
        event_config: Event configuration dictionary from config.py
    
    Returns:
        Dictionary of rainfall features
    """
    from .config import CLIMATIC
    
    BASE_DIR = Path("/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING")
    extractor = EventRainfallExtractor(
        BASE_DIR / "CLIMATIC DATAS/Zamboonga_CHIRPS_Daily_2021_2026.csv",
        BASE_DIR / "CLIMATIC DATAS/Zamboonga_GPM_V07_Daily_2021_2026.csv"
    )
    
    return extractor.extract_event_features(event_config)


def get_all_event_rainfall_features() -> dict:
    """
    Get rainfall features for ALL flood events from config
    
    Returns:
        Dictionary of all rainfall features
    """
    from .config import FLOOD_EVENTS
    
    all_features = {}
    for event_key, event_config in FLOOD_EVENTS.items():
        features = get_event_rainfall_features(event_config)
        all_features.update(features)
    
    return all_features


def get_active_event_rainfall_features() -> dict:
    """
    Get rainfall features for the ACTIVE event only
    
    Returns:
        Dictionary of rainfall features for the active event
    """
    from .config import FLOOD_EVENTS, ACTIVE_EVENT
    
    if ACTIVE_EVENT not in FLOOD_EVENTS:
        logger.warning(f"Active event '{ACTIVE_EVENT}' not found in FLOOD_EVENTS")
        return {}
    
    return get_event_rainfall_features(FLOOD_EVENTS[ACTIVE_EVENT])