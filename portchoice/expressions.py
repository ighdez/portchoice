"""Portchoice expressions"""
class Attribute:
    def __init__(self, name: str, alt: str, levels: list):
        
        # Set parameters
        self.name = name
        self.alt = alt
        self.levels = levels
        self.alt_name = alt + '_' + name