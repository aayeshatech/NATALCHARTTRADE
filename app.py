# app.py - Enhanced Streamlit Astrological Trading Analysis App

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from datetime import datetime, timedelta
import pytz
import pandas as pd
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.dates as mdates

# ======================
# MARKET CONFIGURATIONS
# ======================
MARKET_CONFIGS = {
    'NIFTY': {
        'timezone': 'Asia/Kolkata',
        'default_hours': (9, 15, 15, 30),  # 9:15 AM to 3:30 PM
        'price_scale': 100,
        'volatility_factor': 0.0015,
        'trading_days': 'mon-fri'  # Monday to Friday
    },
    'BANKNIFTY': {
        'timezone': 'Asia/Kolkata',
        'default_hours': (9, 15, 15, 30),
        'price_scale': 100,
        'volatility_factor': 0.002,
        'trading_days': 'mon-fri'
    },
    'CRUDE': {
        'timezone': 'America/New_York',
        'default_hours': (5, 0, 23, 55),
        'price_scale': 10,
        'volatility_factor': 0.003,
        'trading_days': 'mon-fri'
    },
    'SILVER': {
        'timezone': 'America/New_York',
        'default_hours': (5, 0, 23, 55),
        'price_scale': 1,
        'volatility_factor': 0.002,
        'trading_days': 'mon-fri'
    },
    'GOLD': {
        'timezone': 'America/New_York',
        'default_hours': (5, 0, 23, 55),
        'price_scale': 1,
        'volatility_factor': 0.002,
        'trading_days': 'mon-fri'
    }
}

# ======================
# ASTROLOGICAL TRADING FRAMEWORK
# ======================

class AstroTradingFramework:
    def __init__(self, symbol, price, date=None, market_hours=None):
        self.symbol = symbol.upper()
        self.price = price
        self.date = date if date else datetime.now()
        
        # Get market configuration
        if self.symbol in MARKET_CONFIGS:
            self.config = MARKET_CONFIGS[self.symbol]
        else:
            self.config = MARKET_CONFIGS['GOLD']
        
        # Use custom market hours if provided, otherwise use default
        if market_hours:
            self.market_hours = market_hours
        else:
            self.market_hours = self.config['default_hours']
        
        # Default planetary positions
        self.planetary_positions = self._get_planetary_positions()
        
        # Calculate price positions
        self.price_positions = self._calculate_price_positions()
        
        # Initialize analysis results
        self.aspects = None
        self.signals = None
        self.intraday_analysis = None
        self.weekly_analysis = None
        self.monthly_analysis = None
    
    def _get_planetary_positions(self):
        """Get planetary positions for the specified date"""
        try:
            # In a real implementation, you would use ephem or another library
            # For this example, we'll use pre-calculated positions
            return {
                'Sun': 132.5,    # Leo
                'Moon': 45.2,    # Taurus
                'Mercury': 155.8, # Virgo
                'Venus': 210.3,  # Libra
                'Mars': 85.7,    # Gemini
                'Jupiter': 55.4,  # Taurus
                'Saturn': 355.2,  # Pisces
                'Uranus': 27.8,   # Aries
                'Neptune': 352.1, # Pisces
                'Pluto': 298.5    # Capricorn
            }
        except Exception as e:
            print(f"Error calculating planetary positions: {e}")
            # Fallback positions
            return {
                'Sun': 0, 'Moon': 0, 'Mercury': 0, 'Venus': 0, 'Mars': 0,
                'Jupiter': 0, 'Saturn': 0, 'Uranus': 0, 'Neptune': 0, 'Pluto': 0
            }
    
    def _calculate_price_positions(self):
        """Calculate price positions using three methods"""
        positions = {}
        
        # Method 1: Simple Modulo 12
        method1_pos = self.price % 12
        positions['Method 1'] = {
            'angle': method1_pos * 30,
            'zodiac': self._get_zodiac_from_angle(method1_pos * 30),
            'description': f"Position {method1_pos}"
        }
        
        # Method 2: Degree Conversion
        method2_angle = self.price % 360
        positions['Method 2'] = {
            'angle': method2_angle,
            'zodiac': self._get_zodiac_from_angle(method2_angle),
            'description': f"{method2_angle}°"
        }
        
        # Method 3: Scaled Modulo
        scale_factor = self.config['price_scale']
        scaled_price = self.price % scale_factor
        method3_pos = (scaled_price / scale_factor) * 12
        positions['Method 3'] = {
            'angle': method3_pos * 30,
            'zodiac': self._get_zodiac_from_angle(method3_pos * 30),
            'description': f"Scaled Position {method3_pos:.1f}"
        }
        
        return positions
    
    def _get_zodiac_from_angle(self, angle):
        """Convert angle to zodiac sign"""
        zodiacs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                   'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        index = int(angle // 30) % 12
        return zodiacs[index]
    
    def _calculate_aspect(self, angle1, angle2):
        """Calculate the angular distance between two points"""
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _check_aspects(self, orb=5):
        """Check aspects between price points and planets"""
        aspects = {
            'Method 1': [],
            'Method 2': [],
            'Method 3': []
        }
        
        aspect_types = {
            'Conjunction': 0,
            'Sextile': 60,
            'Square': 90,
            'Trine': 120,
            'Opposition': 180
        }
        
        for method, price_data in self.price_positions.items():
            price_angle = price_data['angle']
            
            for planet, planet_angle in self.planetary_positions.items():
                aspect_diff = self._calculate_aspect(price_angle, planet_angle)
                
                for aspect_name, aspect_angle in aspect_types.items():
                    if abs(aspect_diff - aspect_angle) <= orb:
                        aspects[method].append({
                            'planet': planet,
                            'aspect': aspect_name,
                            'orb': abs(aspect_diff - aspect_angle),
                            'strength': self._get_aspect_strength(aspect_name)
                        })
        
        return aspects
    
    def _get_aspect_strength(self, aspect_name):
        """Get strength rating for aspect type"""
        strengths = {
            'Conjunction': 5,
            'Opposition': 5,
            'Square': 4,
            'Trine': 3,
            'Sextile': 2
        }
        return strengths.get(aspect_name, 1)
    
    def generate_signals(self):
        """Generate trading signals based on aspects"""
        if not self.aspects:
            self.aspects = self._check_aspects()
        
        signals = {
            'buy': [],
            'sell': [],
            'caution': [],
            'opportunity': []
        }
        
        # Planet classifications
        benefics = ['Venus', 'Jupiter']
        malefics = ['Mars', 'Saturn']
        
        for method, method_aspects in self.aspects.items():
            for aspect_data in method_aspects:
                planet = aspect_data['planet']
                aspect = aspect_data['aspect']
                strength = aspect_data['strength']
                
                # Buy signals
                if aspect in ['Conjunction', 'Trine', 'Sextile'] and planet in benefics:
                    signals['buy'].append({
                        'method': method,
                        'planet': planet,
                        'aspect': aspect,
                        'strength': strength,
                        'reason': f"{planet} {aspect} to {method}"
                    })
                
                # Sell signals
                elif aspect in ['Conjunction', 'Opposition', 'Square'] and planet in malefics:
                    signals['sell'].append({
                        'method': method,
                        'planet': planet,
                        'aspect': aspect,
                        'strength': strength,
                        'reason': f"{planet} {aspect} to {method}"
                    })
                
                # Caution signals
                elif aspect in ['Opposition', 'Square']:
                    signals['caution'].append({
                        'method': method,
                        'planet': planet,
                        'aspect': aspect,
                        'strength': strength,
                        'reason': f"{planet} {aspect} to {method}"
                    })
                
                # Opportunity signals
                elif aspect in ['Trine', 'Sextile']:
                    signals['opportunity'].append({
                        'method': method,
                        'planet': planet,
                        'aspect': aspect,
                        'strength': strength,
                        'reason': f"{planet} {aspect} to {method}"
                    })
        
        self.signals = signals
        return signals
    
    def generate_intraday_analysis(self):
        """Generate intraday analysis with price predictions"""
        start_hour, start_minute, end_hour, end_minute = self.market_hours
        
        # Create time grid for analysis (hourly)
        hours = []
        current_time = self.date.replace(hour=start_hour, minute=start_minute)
        end_time = self.date.replace(hour=end_hour, minute=end_minute)
        
        while current_time <= end_time:
            hours.append(current_time.hour + current_time.minute/60)
            current_time += timedelta(hours=1)
        
        analysis = []
        
        for hour in hours:
            # Get planetary positions for this hour
            planet_positions = self.planetary_positions.copy()
            
            # Simulate planetary movement
            for planet in planet_positions:
                speed = {
                    'Sun': 0.04, 'Moon': 0.5, 'Mercury': 0.05, 'Venus': 0.03,
                    'Mars': 0.03, 'Jupiter': 0.01, 'Saturn': 0.005,
                    'Uranus': 0.003, 'Neptune': 0.002, 'Pluto': 0.001
                }.get(planet, 0.01)
                
                planet_positions[planet] = (planet_positions[planet] + speed * hour) % 360
            
            # Check aspects to price points
            price_aspects = {}
            for method, pos_data in self.price_positions.items():
                price_aspects[method] = []
                for planet, planet_angle in planet_positions.items():
                    aspect_diff = self._calculate_aspect(pos_data['angle'], planet_angle)
                    
                    aspect_types = {
                        'Conjunction': 0, 'Sextile': 60, 'Square': 90, 'Trine': 120, 'Opposition': 180
                    }
                    
                    for aspect_name, aspect_angle in aspect_types.items():
                        if abs(aspect_diff - aspect_angle) <= 5:  # 5-degree orb
                            price_aspects[method].append({
                                'planet': planet,
                                'aspect': aspect_name,
                                'orb': abs(aspect_diff - aspect_angle)
                            })
            
            # Determine price direction and strength
            price_direction, price_strength = self._predict_price_movement(price_aspects, hour)
            
            # Calculate expected price range
            expected_high, expected_low = self._calculate_price_range(
                self.price, price_direction, price_strength, hour
            )
            
            analysis.append({
                'hour': hour,
                'time': f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
                'price_direction': price_direction,
                'price_strength': price_strength,
                'expected_high': expected_high,
                'expected_low': expected_low
            })
        
        self.intraday_analysis = analysis
        return analysis
    
    def generate_weekly_analysis(self):
        """Generate weekly analysis with price predictions"""
        # Get the start of the week (Monday)
        start_date = self.date - timedelta(days=self.date.weekday())
        
        # Create date grid for the week
        dates = []
        for i in range(7):
            dates.append(start_date + timedelta(days=i))
        
        analysis = []
        
        for date in dates:
            # Skip weekends for stock market symbols
            if self.config['trading_days'] == 'mon-fri' and date.weekday() >= 5:
                continue
            
            # Get planetary positions for this date
            planet_positions = self.planetary_positions.copy()
            
            # Simulate daily planetary movement
            days_diff = (date - self.date).days
            for planet in planet_positions:
                daily_speed = {
                    'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.2, 'Venus': 1.1,
                    'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03,
                    'Uranus': 0.04, 'Neptune': 0.02, 'Pluto': 0.03
                }.get(planet, 0.1)
                
                planet_positions[planet] = (planet_positions[planet] + daily_speed * days_diff) % 360
            
            # Check aspects to price points
            price_aspects = {}
            for method, pos_data in self.price_positions.items():
                price_aspects[method] = []
                for planet, planet_angle in planet_positions.items():
                    aspect_diff = self._calculate_aspect(pos_data['angle'], planet_angle)
                    
                    aspect_types = {
                        'Conjunction': 0, 'Sextile': 60, 'Square': 90, 'Trine': 120, 'Opposition': 180
                    }
                    
                    for aspect_name, aspect_angle in aspect_types.items():
                        if abs(aspect_diff - aspect_angle) <= 5:  # 5-degree orb
                            price_aspects[method].append({
                                'planet': planet,
                                'aspect': aspect_name,
                                'orb': abs(aspect_diff - aspect_angle)
                            })
            
            # Determine price direction and strength
            price_direction, price_strength = self._predict_price_movement(price_aspects, date.hour)
            
            # Calculate expected price range
            expected_high, expected_low = self._calculate_price_range(
                self.price, price_direction, price_strength, date.hour
            )
            
            # Simulate actual price (for demonstration)
            actual_price = (expected_high + expected_low) / 2 + np.random.normal(0, (expected_high - expected_low) * 0.1)
            
            analysis.append({
                'date': date,
                'day_name': date.strftime('%a'),
                'price_direction': price_direction,
                'price_strength': price_strength,
                'expected_high': expected_high,
                'expected_low': expected_low,
                'actual_price': actual_price,
                'planet_positions': planet_positions
            })
        
        self.weekly_analysis = analysis
        return analysis
    
    def generate_monthly_analysis(self):
        """Generate monthly analysis with price predictions"""
        # Get the first day of the month
        start_date = self.date.replace(day=1)
        
        # Get the number of days in the month
        if start_date.month == 12:
            next_month = start_date.replace(year=start_date.year + 1, month=1)
        else:
            next_month = start_date.replace(month=start_date.month + 1)
        
        days_in_month = (next_month - start_date).days
        
        # Create date grid for the month
        dates = []
        for i in range(days_in_month):
            dates.append(start_date + timedelta(days=i))
        
        analysis = []
        
        for date in dates:
            # Skip weekends for stock market symbols
            if self.config['trading_days'] == 'mon-fri' and date.weekday() >= 5:
                continue
            
            # Get planetary positions for this date
            planet_positions = self.planetary_positions.copy()
            
            # Simulate daily planetary movement
            days_diff = (date - self.date).days
            for planet in planet_positions:
                daily_speed = {
                    'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.2, 'Venus': 1.1,
                    'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03,
                    'Uranus': 0.04, 'Neptune': 0.02, 'Pluto': 0.03
                }.get(planet, 0.1)
                
                planet_positions[planet] = (planet_positions[planet] + daily_speed * days_diff) % 360
            
            # Check aspects to price points
            price_aspects = {}
            for method, pos_data in self.price_positions.items():
                price_aspects[method] = []
                for planet, planet_angle in planet_positions.items():
                    aspect_diff = self._calculate_aspect(pos_data['angle'], planet_angle)
                    
                    aspect_types = {
                        'Conjunction': 0, 'Sextile': 60, 'Square': 90, 'Trine': 120, 'Opposition': 180
                    }
                    
                    for aspect_name, aspect_angle in aspect_types.items():
                        if abs(aspect_diff - aspect_angle) <= 5:  # 5-degree orb
                            price_aspects[method].append({
                                'planet': planet,
                                'aspect': aspect_name,
                                'orb': abs(aspect_diff - aspect_angle)
                            })
            
            # Determine price direction and strength
            price_direction, price_strength = self._predict_price_movement(price_aspects, date.hour)
            
            # Calculate expected price range
            expected_high, expected_low = self._calculate_price_range(
                self.price, price_direction, price_strength, date.hour
            )
            
            # Simulate actual price (for demonstration)
            actual_price = (expected_high + expected_low) / 2 + np.random.normal(0, (expected_high - expected_low) * 0.1)
            
            analysis.append({
                'date': date,
                'day': date.day,
                'price_direction': price_direction,
                'price_strength': price_strength,
                'expected_high': expected_high,
                'expected_low': expected_low,
                'actual_price': actual_price,
                'planet_positions': planet_positions
            })
        
        self.monthly_analysis = analysis
        return analysis
    
    def _predict_price_movement(self, price_aspects, hour):
        """Predict price movement based on aspects"""
        bullish_score = 0
        bearish_score = 0
        
        # Analyze price point aspects
        for method, aspects in price_aspects.items():
            for aspect in aspects:
                planet = aspect['planet']
                aspect_type = aspect['aspect']
                
                # Benefic planets in harmonious aspects
                if planet in ['Venus', 'Jupiter'] and aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                    bullish_score += 3
                # Malefic planets in challenging aspects
                elif planet in ['Mars', 'Saturn'] and aspect_type in ['Opposition', 'Square']:
                    bearish_score += 3
                # Moon aspects
                elif planet == 'Moon':
                    if aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                        bullish_score += 1
                    else:
                        bearish_score += 1
        
        # Consider time of day
        if 9 <= hour <= 11:
            bullish_score += 1
        elif 14 <= hour <= 16:
            bullish_score += 1
        elif 20 <= hour <= 22:
            bearish_score += 1
        
        # Determine direction and strength
        if bullish_score > bearish_score:
            direction = 'Bullish'
            strength = min(5, (bullish_score - bearish_score) // 2 + 1)
        elif bearish_score > bullish_score:
            direction = 'Bearish'
            strength = min(5, (bearish_score - bullish_score) // 2 + 1)
        else:
            direction = 'Neutral'
            strength = 1
        
        return direction, strength
    
    def _calculate_price_range(self, current_price, direction, strength, hour):
        """Calculate expected price range"""
        base_volatility = self.config['volatility_factor']
        volatility_multiplier = 0.5 + (strength * 0.3)
        
        if hour in [9, 10, 14, 15, 20, 21]:
            volatility_multiplier *= 1.5
        
        price_change = current_price * base_volatility * volatility_multiplier
        
        if direction == 'Bullish':
            expected_high = current_price + price_change
            expected_low = current_price - (price_change * 0.3)
        elif direction == 'Bearish':
            expected_high = current_price + (price_change * 0.3)
            expected_low = current_price - price_change
        else:
            expected_high = current_price + (price_change * 0.5)
            expected_low = current_price - (price_change * 0.5)
        
        return round(expected_high, 2), round(expected_low, 2)
    
    def create_chart(self, show_aspects=True):
        """Create the astrological chart visualization"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Zodiac signs and houses
        zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                        'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        houses = list(range(1, 13))
        
        # Set up the chart
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 12)
        
        # Draw zodiac segments
        for i, sign in enumerate(zodiac_signs):
            angle = i * 30
            ax.text(angle, 11.5, sign, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.plot([angle, angle], [0, 12], 'k-', alpha=0.3, linewidth=1)
        
        # Draw houses
        for i in range(12):
            ax.plot([i*30, i*30], [0, 10], 'k--', alpha=0.5)
            ax.text(i*30, 9, str(houses[i]), ha='center', va='center', fontsize=10)
        
        # Plot planets
        planet_symbols = {
            'Sun': '☉', 'Moon': '☽', 'Mercury': '☿', 'Venus': '♀', 'Mars': '♂',
            'Jupiter': '♃', 'Saturn': '♄', 'Uranus': '♅', 'Neptune': '♆', 'Pluto': '♇'
        }
        
        planet_colors = {
            'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray', 'Venus': 'lightgreen',
            'Mars': 'red', 'Jupiter': 'orange', 'Saturn': 'darkgoldenrod',
            'Uranus': 'lightblue', 'Neptune': 'darkblue', 'Pluto': 'darkred'
        }
        
        for planet, angle in self.planetary_positions.items():
            ax.plot([angle], [8], 'o', color=planet_colors[planet], markersize=12, alpha=0.8)
            ax.text(angle, 8.5, planet_symbols[planet], ha='center', va='center', 
                    fontsize=16, color=planet_colors[planet], fontweight='bold')
            ax.text(angle, 7.5, planet, ha='center', va='center', fontsize=10, color='black')
        
        # Plot price points
        method_colors = {'Method 1': 'red', 'Method 2': 'green', 'Method 3': 'blue'}
        for method, pos_data in self.price_positions.items():
            angle = pos_data['angle']
            ax.plot([angle], [5], 'o', color=method_colors[method], markersize=15)
            ax.text(angle, 5.5, str(self.price), color=method_colors[method], 
                    ha='center', fontweight='bold', fontsize=12)
            ax.text(angle, 4.5, method, color=method_colors[method], 
                    ha='center', fontsize=10)
        
        # Draw aspects
        if show_aspects and self.aspects:
            aspect_colors = {
                'Conjunction': 'black', 'Opposition': 'red', 'Trine': 'blue',
                'Square': 'green', 'Sextile': 'purple'
            }
            
            for method, method_aspects in self.aspects.items():
                price_angle = self.price_positions[method]['angle']
                for aspect in method_aspects:
                    planet = aspect['planet']
                    aspect_type = aspect['aspect']
                    planet_angle = self.planetary_positions[planet]
                    
                    ax.plot([price_angle, planet_angle], [5, 8], 
                           color=aspect_colors[aspect_type], linewidth=2, alpha=0.7)
        
        # Add legends
        price_legend = []
        for method, color in method_colors.items():
            price_legend.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color, markersize=10, label=method))
        
        aspect_legend = []
        if show_aspects:
            aspect_colors = {
                'Conjunction': 'black', 'Opposition': 'red', 'Trine': 'blue',
                'Square': 'green', 'Sextile': 'purple'
            }
            for aspect, color in aspect_colors.items():
                aspect_legend.append(plt.Line2D([0], [0], color=color, linewidth=2, label=aspect))
        
        price_legend = ax.legend(handles=price_legend, loc='upper right', 
                                bbox_to_anchor=(1.3, 1.15), fontsize=10)
        
        if aspect_legend:
            aspect_legend = ax.legend(handles=aspect_legend, loc='upper right', 
                                     bbox_to_anchor=(1.3, 0.85), fontsize=10, title='Aspects')
            ax.add_artist(price_legend)
        
        # Add title
        plt.title(f'{self.symbol} Price ({self.price}) with Planetary Positions and Aspects\n'
                  f'Financial Astrology Analysis for {self.date.strftime("%Y-%m-%d")}', 
                  fontsize=18, pad=30, fontweight='bold')
        
        # Remove grid and axis labels
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def create_intraday_chart(self):
        """Create intraday price prediction chart"""
        if not self.intraday_analysis:
            self.generate_intraday_analysis()
        
        # Prepare data
        hours = [a['hour'] for a in self.intraday_analysis]
        times = [a['time'] for a in self.intraday_analysis]
        expected_highs = [a['expected_high'] for a in self.intraday_analysis]
        expected_lows = [a['expected_low'] for a in self.intraday_analysis]
        directions = [a['price_direction'] for a in self.intraday_analysis]
        strengths = [a['price_strength'] for a in self.intraday_analysis]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.fill_between(hours, expected_lows, expected_highs, alpha=0.3, color='gold', label='Expected Range')
        ax1.plot(hours, expected_highs, 'g-', linewidth=2, label='Expected High')
        ax1.plot(hours, expected_lows, 'r-', linewidth=2, label='Expected Low')
        
        # Mark key events
        for i, (hour, direction, strength) in enumerate(zip(hours, directions, strengths)):
            if direction == 'Bullish':
                ax1.scatter(hour, expected_highs[i], color='green', s=strength*20, alpha=0.7)
            elif direction == 'Bearish':
                ax1.scatter(hour, expected_lows[i], color='red', s=strength*20, alpha=0.7)
        
        # Formatting
        ax1.set_title(f'{self.symbol} Intraday Price Prediction - {self.date.strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_xticks(hours[::2])
        ax1.set_xticklabels([times[i] for i in range(0, len(times), 2)], rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Direction/Strength chart
        colors = []
        for direction in directions:
            if direction == 'Bullish':
                colors.append('green')
            elif direction == 'Bearish':
                colors.append('red')
            else:
                colors.append('gray')
        
        ax2.bar(hours, strengths, color=colors, alpha=0.7)
        ax2.set_ylabel('Strength', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_xticks(hours[::2])
        ax2.set_xticklabels([times[i] for i in range(0, len(times), 2)], rotation=45)
        ax2.set_ylim(0, 6)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_weekly_chart(self):
        """Create weekly price prediction chart with planetary transits"""
        if not self.weekly_analysis:
            self.generate_weekly_analysis()
        
        # Prepare data
        dates = [a['date'] for a in self.weekly_analysis]
        day_names = [a['day_name'] for a in self.weekly_analysis]
        expected_highs = [a['expected_high'] for a in self.weekly_analysis]
        expected_lows = [a['expected_low'] for a in self.weekly_analysis]
        actual_prices = [a['actual_price'] for a in self.weekly_analysis]
        directions = [a['price_direction'] for a in self.weekly_analysis]
        strengths = [a['price_strength'] for a in self.weekly_analysis]
        
        # Get planetary positions for the week (Sun and Moon for example)
        sun_positions = [a['planet_positions']['Sun'] for a in self.weekly_analysis]
        moon_positions = [a['planet_positions']['Moon'] for a in self.weekly_analysis]
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()
        
        # Plot price data
        ax1.fill_between(dates, expected_lows, expected_highs, alpha=0.3, color='gold', label='Expected Range')
        ax1.plot(dates, expected_highs, 'g--', linewidth=1, alpha=0.7, label='Expected High')
        ax1.plot(dates, expected_lows, 'r--', linewidth=1, alpha=0.7, label='Expected Low')
        ax1.plot(dates, actual_prices, 'b-', linewidth=2, label='Actual Price')
        
        # Mark key events
        for i, (date, direction, strength) in enumerate(zip(dates, directions, strengths)):
            if direction == 'Bullish':
                ax1.scatter(date, actual_prices[i], color='green', s=strength*30, alpha=0.7, zorder=5)
            elif direction == 'Bearish':
                ax1.scatter(date, actual_prices[i], color='red', s=strength*30, alpha=0.7, zorder=5)
        
        # Plot planetary positions on secondary axis
        ax2.plot(dates, sun_positions, 'o-', color='gold', linewidth=2, markersize=8, label='Sun Position')
        ax2.plot(dates, moon_positions, 'o-', color='silver', linewidth=2, markersize=8, label='Moon Position')
        
        # Formatting
        ax1.set_title(f'{self.symbol} Weekly Price Prediction with Planetary Transits\n'
                     f'Week of {self.date.strftime("%Y-%m-%d")}', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price', fontsize=12, color='blue')
        ax2.set_ylabel('Planetary Position (degrees)', fontsize=12, color='orange')
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(DateFormatter('%a %m-%d'))
        ax1.xaxis.set_major_locator(DayLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Grid and legends
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_monthly_chart(self):
        """Create monthly price prediction chart with planetary transits"""
        if not self.monthly_analysis:
            self.generate_monthly_analysis()
        
        # Prepare data
        dates = [a['date'] for a in self.monthly_analysis]
        days = [a['day'] for a in self.monthly_analysis]
        expected_highs = [a['expected_high'] for a in self.monthly_analysis]
        expected_lows = [a['expected_low'] for a in self.monthly_analysis]
        actual_prices = [a['actual_price'] for a in self.monthly_analysis]
        directions = [a['price_direction'] for a in self.monthly_analysis]
        strengths = [a['price_strength'] for a in self.monthly_analysis]
        
        # Get planetary positions for the month (Sun and Moon for example)
        sun_positions = [a['planet_positions']['Sun'] for a in self.monthly_analysis]
        moon_positions = [a['planet_positions']['Moon'] for a in self.monthly_analysis]
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = ax1.twinx()
        
        # Plot price data
        ax1.fill_between(dates, expected_lows, expected_highs, alpha=0.3, color='gold', label='Expected Range')
        ax1.plot(dates, expected_highs, 'g--', linewidth=1, alpha=0.7, label='Expected High')
        ax1.plot(dates, expected_lows, 'r--', linewidth=1, alpha=0.7, label='Expected Low')
        ax1.plot(dates, actual_prices, 'b-', linewidth=2, label='Actual Price')
        
        # Mark key events
        for i, (date, direction, strength) in enumerate(zip(dates, directions, strengths)):
            if direction == 'Bullish':
                ax1.scatter(date, actual_prices[i], color='green', s=strength*20, alpha=0.7, zorder=5)
            elif direction == 'Bearish':
                ax1.scatter(date, actual_prices[i], color='red', s=strength*20, alpha=0.7, zorder=5)
        
        # Plot planetary positions on secondary axis
        ax2.plot(dates, sun_positions, 'o-', color='gold', linewidth=2, markersize=6, label='Sun Position')
        ax2.plot(dates, moon_positions, 'o-', color='silver', linewidth=2, markersize=6, label='Moon Position')
        
        # Formatting
        ax1.set_title(f'{self.symbol} Monthly Price Prediction with Planetary Transits\n'
                     f'{self.date.strftime("%B %Y")}', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price', fontsize=12, color='blue')
        ax2.set_ylabel('Planetary Position (degrees)', fontsize=12, color='orange')
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Grid and legends
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def get_analysis_dataframe(self, analysis_type='intraday'):
        """Return analysis results as a DataFrame"""
        if analysis_type == 'intraday' and self.intraday_analysis:
            return pd.DataFrame(self.intraday_analysis)
        elif analysis_type == 'weekly' and self.weekly_analysis:
            df = pd.DataFrame(self.weekly_analysis)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            return df
        elif analysis_type == 'monthly' and self.monthly_analysis:
            df = pd.DataFrame(self.monthly_analysis)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            return df
        return pd.DataFrame()

# ======================
# STREAMLIT APP
# ======================

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Astrological Trading Analysis",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #43A047;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .chart-container {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>✨ Astrological Trading Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Planetary influences on financial markets</p>", unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.header("Analysis Parameters")
    
    # Symbol selection
    symbol_options = ['NIFTY', 'BANKNIFTY', 'CRUDE', 'SILVER', 'GOLD', 'Custom']
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbol_options)
    
    if selected_symbol == 'Custom':
        custom_symbol = st.sidebar.text_input("Enter Custom Symbol")
        symbol = custom_symbol if custom_symbol else 'CUSTOM'
    else:
        symbol = selected_symbol
    
    # Price input
    price = st.sidebar.number_input("Enter Current Price", min_value=0.0, value=100.0, step=0.01)
    
    # Date selection
    selected_date = st.sidebar.date_input("Select Date", datetime.now())
    
    # Analysis type
    analysis_type = st.sidebar.radio("Analysis Type", ["Daily", "Intraday", "Weekly", "Monthly"])
    
    # Custom market hours for intraday
    market_hours = None
    if analysis_type == "Intraday":
        use_custom_hours = st.sidebar.checkbox("Use Custom Market Hours")
        
        if use_custom_hours:
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                start_hour = st.selectbox("Start Hour", list(range(0, 24)), index=9)
                start_minute = st.selectbox("Start Minute", [0, 15, 30, 45], index=1)
            
            with col2:
                end_hour = st.selectbox("End Hour", list(range(0, 24)), index=15)
                end_minute = st.selectbox("End Minute", [0, 15, 30, 45], index=2)
            
            market_hours = (start_hour, start_minute, end_hour, end_minute)
            st.sidebar.write(f"Custom Hours: {start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}")
    
    # Analyze button
    analyze_button = st.sidebar.button("Analyze", key="analyze")
    
    # Main content area
    if analyze_button:
        # Create analysis object
        analysis = AstroTradingFramework(
            symbol=symbol, 
            price=price, 
            date=selected_date,
            market_hours=market_hours
        )
        
        # Generate signals
        analysis.generate_signals()
        
        # Generate analysis based on type
        if analysis_type == "Intraday":
            analysis.generate_intraday_analysis()
        elif analysis_type == "Weekly":
            analysis.generate_weekly_analysis()
        elif analysis_type == "Monthly":
            analysis.generate_monthly_analysis()
        
        # Display results
        st.header(f"{symbol} Analysis Results")
        
        # Display planetary positions
        st.subheader("Planetary Positions")
        planet_data = []
        for planet, angle in analysis.planetary_positions.items():
            zodiac = analysis._get_zodiac_from_angle(angle)
            planet_data.append({"Planet": planet, "Angle": f"{angle:.2f}°", "Zodiac": zodiac})
        
        planet_df = pd.DataFrame(planet_data)
        st.dataframe(planet_df, use_container_width=True)
        
        # Display price positions
        st.subheader("Price Positions")
        price_data = []
        for method, data in analysis.price_positions.items():
            price_data.append({
                "Method": method, 
                "Description": data['description'], 
                "Angle": f"{data['angle']:.2f}°", 
                "Zodiac": data['zodiac']
            })
        
        price_df = pd.DataFrame(price_data)
        st.dataframe(price_df, use_container_width=True)
        
        # Display aspects
        st.subheader("Aspects")
        if analysis.aspects:
            aspect_data = []
            for method, aspects in analysis.aspects.items():
                for aspect in aspects:
                    aspect_data.append({
                        "Method": method,
                        "Planet": aspect['planet'],
                        "Aspect": aspect['aspect'],
                        "Orb": f"{aspect['orb']:.2f}°",
                        "Strength": aspect['strength']
                    })
            
            aspect_df = pd.DataFrame(aspect_data)
            st.dataframe(aspect_df, use_container_width=True)
        else:
            st.info("No significant aspects found")
        
        # Display trading signals
        st.subheader("Trading Signals")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Buy Signals")
            if analysis.signals['buy']:
                for signal in analysis.signals['buy']:
                    st.success(f"{signal['reason']} (Strength: {signal['strength']}/5)")
            else:
                st.info("No buy signals")
        
        with col2:
            st.markdown("#### Sell Signals")
            if analysis.signals['sell']:
                for signal in analysis.signals['sell']:
                    st.error(f"{signal['reason']} (Strength: {signal['strength']}/5)")
            else:
                st.info("No sell signals")
        
        with col3:
            st.markdown("#### Caution Signals")
            if analysis.signals['caution']:
                for signal in analysis.signals['caution']:
                    st.warning(f"{signal['reason']} (Strength: {signal['strength']}/5)")
            else:
                st.info("No caution signals")
        
        with col4:
            st.markdown("#### Opportunity Signals")
            if analysis.signals['opportunity']:
                for signal in analysis.signals['opportunity']:
                    st.info(f"{signal['reason']} (Strength: {signal['strength']}/5)")
            else:
                st.info("No opportunity signals")
        
        # Display analysis data
        if analysis_type in ["Intraday", "Weekly", "Monthly"]:
            st.subheader(f"{analysis_type} Analysis")
            
            # Get DataFrame
            df = analysis.get_analysis_dataframe(analysis_type.lower())
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Display key periods
                if analysis_type == "Intraday":
                    key_periods = [a for a in analysis.intraday_analysis if a['price_strength'] >= 3]
                elif analysis_type == "Weekly":
                    key_periods = [a for a in analysis.weekly_analysis if a['price_strength'] >= 3]
                elif analysis_type == "Monthly":
                    key_periods = [a for a in analysis.monthly_analysis if a['price_strength'] >= 3]
                
                if key_periods:
                    st.subheader(f"Key {analysis_type} Periods")
                    for period in key_periods:
                        if analysis_type == "Intraday":
                            time_key = period['time']
                        elif analysis_type == "Weekly":
                            time_key = f"{period['day_name']} ({period['date'].strftime('%m-%d')})"
                        elif analysis_type == "Monthly":
                            time_key = f"Day {period['day']} ({period['date'].strftime('%m-%d')})"
                        
                        if period['price_direction'] == 'Bullish':
                            st.success(f"{time_key}: {period['price_direction']} (Strength {period['price_strength']})")
                        elif period['price_direction'] == 'Bearish':
                            st.error(f"{time_key}: {period['price_direction']} (Strength {period['price_strength']})")
                        else:
                            st.info(f"{time_key}: {period['price_direction']} (Strength {period['price_strength']})")
        
        # Display charts
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Astrological Chart")
            fig1 = analysis.create_chart()
            st.pyplot(fig1)
        
        with col2:
            if analysis_type == "Intraday":
                st.markdown("#### Intraday Price Prediction")
                fig2 = analysis.create_intraday_chart()
                st.pyplot(fig2)
            elif analysis_type == "Weekly":
                st.markdown("#### Weekly Price Prediction with Transits")
                fig2 = analysis.create_weekly_chart()
                st.pyplot(fig2)
            elif analysis_type == "Monthly":
                st.markdown("#### Monthly Price Prediction with Transits")
                fig2 = analysis.create_monthly_chart()
                st.pyplot(fig2)
            else:
                st.info("Additional charts not available for daily analysis")
        
        # Add disclaimer
        st.markdown("---")
        st.markdown("**Disclaimer**: This tool is for educational purposes only. Astrological analysis should not be used as the sole basis for trading decisions. Always conduct thorough research and consider multiple factors before making financial decisions.")

if __name__ == "__main__":
    main()
