"""
Connectors module for dynamic data sources.

This module provides connectors for retrieving data from sources that are not
file-based, such as Outlook email clients, databases, APIs, etc.
"""

from .outlook_connector import OutlookConnector, OutlookConfig

__all__ = ["OutlookConnector", "OutlookConfig"]
