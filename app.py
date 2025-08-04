import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from datetime import datetime, timedelta
import pytz

# ======================
# ASTROLOGICAL TRADING FRAMEWORK
# ======================

class AstroTradingFramework:
    def __init__(self, symbol, price, date=None, timezone='America/New_York'):
        """
        Initialize the framework for a specific symbol and price
        
        Parameters:
        - symbol: Trading symbol (e.g., 'GC=F' for Gold, 'ES=F' for S&P 500)
        - price: Current or closing price
        - date: Date for analysis (default: today)
        - timezone: Market timezone (default: US Eastern)
        """
        self.symbol = symbol
        self.price = price
        self.date = date if date else datetime.now()
        self.timezone = timezone
        
        # Default planetary positions (will be updated with actual calculation)
        self.planetary_positions = self._get_planetary_positions()
        
        # Calculate price positions
        self.price_positions = self._calculate_price_positions()
        
        # Initialize analysis results
        self.aspects = None
        self.signals = None
        self.intraday_analysis = None
    
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
        
        # Method 3: Scaled Modulo (adjust scale based on typical price range)
        # This is a simplified approach - in practice, you'd adjust the scale based on the asset
        scale_factor = 1000 if self.price > 1000 else 100
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
    
    def generate_intraday_analysis(self, market_open=None, market_close=None):
        """
        Generate intraday analysis with price predictions
        
        Parameters:
        - market_open: Market open time (default: 5:00 AM)
        - market_close: Market close time (default: 11:55 PM)
        """
        if not market_open:
            market_open = self.date.replace(hour=5, minute=0)
        if not market_close:
            market_close = self.date.replace(hour=23, minute=55)
        
        # Create time grid for analysis (hourly)
        hours = []
        current_time = market_open
        while current_time <= market_close:
            hours.append(current_time.hour + current_time.minute/60)
            current_time += timedelta(hours=1)
        
        analysis = []
        
        for hour in hours:
            # Get planetary positions for this hour (simplified - in practice would calculate more precisely)
            planet_positions = self.planetary_positions.copy()
            
            # Simulate planetary movement (simplified)
            for planet in planet_positions:
                # Planets move at different speeds
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
    
    def _predict_price_movement(self, price_aspects, hour):
        """Predict price movement based on aspects"""
        # Score for bullish/bearish influences
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
                # Moon aspects (moderate influence)
                elif planet == 'Moon':
                    if aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                        bullish_score += 1
                    else:
                        bearish_score += 1
        
        # Consider time of day (simplified)
        if 9 <= hour <= 11:  # Morning strength
            bullish_score += 1
        elif 14 <= hour <= 16:  # Afternoon strength
            bullish_score += 1
        elif 20 <= hour <= 22:  # Evening strength
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
        """Calculate expected price range for the hour"""
        # Base volatility factor
        base_volatility = 0.002  # 0.2% of price
        
        # Adjust based on strength
        volatility_multiplier = 0.5 + (strength * 0.3)
        
        # Time-based adjustment (higher volatility during key hours)
        if hour in [9, 10, 14, 15, 20, 21]:  # Key trading hours
            volatility_multiplier *= 1.5
        
        # Calculate range
        price_change = current_price * base_volatility * volatility_multiplier
        
        if direction == 'Bullish':
            expected_high = current_price + price_change
            expected_low = current_price - (price_change * 0.3)
        elif direction == 'Bearish':
            expected_high = current_price + (price_change * 0.3)
            expected_low = current_price - price_change
        else:  # Neutral
            expected_high = current_price + (price_change * 0.5)
            expected_low = current_price - (price_change * 0.5)
        
        return round(expected_high, 2), round(expected_low, 2)
    
    def create_chart(self, show_aspects=True, show_intraday=False):
        """Create the astrological chart visualization"""
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, polar=True)
        
        # Zodiac signs and houses
        zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                        'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        houses = list(range(1, 13))
        
        # Set up the chart
        ax.set_theta_zero_location('N')  # 0 degrees at top
        ax.set_theta_direction(-1)  # Clockwise direction
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
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
        ax1.set_xticks(hours[::2])  # Show every other hour
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
    
    def print_analysis(self):
        """Print the complete analysis"""
        print(f"=== {self.symbol} ASTROLOGICAL TRADING ANALYSIS ===\n")
        print(f"Date: {self.date.strftime('%Y-%m-%d')}")
        print(f"Price: {self.price}\n")
        
        # Planetary positions
        print("1. Planetary Positions:")
        for planet, angle in self.planetary_positions.items():
            zodiac = self._get_zodiac_from_angle(angle)
            print(f"   {planet}: {angle:.2f}° {zodiac}")
        
        # Price positions
        print("\n2. Price Positions:")
        for method, data in self.price_positions.items():
            print(f"   {method}: {data['description']} at {data['angle']:.2f}° {data['zodiac']}")
        
        # Aspects
        if not self.aspects:
            self.aspects = self._check_aspects()
        
        print("\n3. Aspects:")
        for method, method_aspects in self.aspects.items():
            print(f"\n   {method} aspects:")
            if method_aspects:
                for aspect in method_aspects:
                    print(f"     {aspect['planet']} {aspect['aspect']} (orb: {aspect['orb']:.2f}°)")
            else:
                print("     No significant aspects")
        
        # Trading signals
        if not self.signals:
            self.signals = self.generate_signals()
        
        print("\n4. Trading Signals:")
        print("\n   BUY SIGNALS:")
        for signal in self.signals['buy']:
            print(f"     - {signal['reason']} (Strength: {signal['strength']}/5)")
        
        print("\n   SELL SIGNALS:")
        for signal in self.signals['sell']:
            print(f"     - {signal['reason']} (Strength: {signal['strength']}/5)")
        
        print("\n   CAUTION SIGNALS:")
        for signal in self.signals['caution']:
            print(f"     - {signal['reason']} (Strength: {signal['strength']}/5)")
        
        print("\n   OPPORTUNITY SIGNALS:")
        for signal in self.signals['opportunity']:
            print(f"     - {signal['reason']} (Strength: {signal['strength']}/5)")
        
        # Intraday analysis
        if self.intraday_analysis:
            print("\n5. Intraday Price Predictions:")
            print("   Time     | Direction | Strength | Expected High | Expected Low")
            print("   " + "-"*60)
            
            for a in self.intraday_analysis:
                print(f"   {a['time']} | {a['price_direction']:9} | {a['price_strength']:8} | {a['expected_high']:13.2f} | {a['expected_low']:12.2f}")
            
            print("\n6. Key Trading Hours:")
            key_hours = []
            for a in self.intraday_analysis:
                if a['price_strength'] >= 3:
                    key_hours.append(a)
            
            if key_hours:
                print("   High-activity periods (Strength 3+):")
                for hour in key_hours:
                    print(f"   - {hour['time']}: {hour['price_direction']} (Strength {hour['price_strength']})")
            else:
                print("   No high-activity periods predicted for today")
        
        print("\n=== ANALYSIS COMPLETE ===")


# ======================
# USAGE EXAMPLES
# ======================

def analyze_gold():
    """Example: Analyze Gold"""
    print("Analyzing Gold (GC=F)...")
    gold = AstroTradingFramework(symbol='GC=F', price=3363, date=datetime(2025, 8, 4))
    gold.generate_signals()
    gold.generate_intraday_analysis()
    gold.print_analysis()
    
    # Create charts
    chart1 = gold.create_chart()
    chart2 = gold.create_intraday_chart()
    
    plt.show()

def analyze_sp500():
    """Example: Analyze S&P 500"""
    print("Analyzing S&P 500 (ES=F)...")
    sp500 = AstroTradingFramework(symbol='ES=F', price=4500, date=datetime(2025, 8, 4))
    sp500.generate_signals()
    sp500.generate_intraday_analysis()
    sp500.print_analysis()
    
    # Create charts
    chart1 = sp500.create_chart()
    chart2 = sp500.create_intraday_chart()
    
    plt.show()

def analyze_eurusd():
    """Example: Analyze EUR/USD"""
    print("Analyzing EUR/USD...")
    eurusd = AstroTradingFramework(symbol='EUR/USD', price=1.0850, date=datetime(2025, 8, 4))
    eurusd.generate_signals()
    eurusd.generate_intraday_analysis()
    eurusd.print_analysis()
    
    # Create charts
    chart1 = eurusd.create_chart()
    chart2 = eurusd.create_intraday_chart()
    
    plt.show()

def analyze_bitcoin():
    """Example: Analyze Bitcoin"""
    print("Analyzing Bitcoin (BTC/USD)...")
    bitcoin = AstroTradingFramework(symbol='BTC/USD', price=65000, date=datetime(2025, 8, 4))
    bitcoin.generate_signals()
    bitcoin.generate_intraday_analysis()
    bitcoin.print_analysis()
    
    # Create charts
    chart1 = bitcoin.create_chart()
    chart2 = bitcoin.create_intraday_chart()
    
    plt.show()

def analyze_custom_symbol():
    """Example: Analyze a custom symbol"""
    symbol = input("Enter symbol (e.g., AAPL, USD/JPY): ")
    price = float(input("Enter current price: "))
    
    # Optional: Enter date (default: today)
    date_input = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
    if date_input:
        year, month, day = map(int, date_input.split('-'))
        date = datetime(year, month, day)
    else:
        date = datetime.now()
    
    print(f"\nAnalyzing {symbol}...")
    custom = AstroTradingFramework(symbol=symbol, price=price, date=date)
    custom.generate_signals()
    custom.generate_intraday_analysis()
    custom.print_analysis()
    
    # Create charts
    chart1 = custom.create_chart()
    chart2 = custom.create_intraday_chart()
    
    plt.show()

# ======================
# MAIN MENU
# ======================

def main():
    """Main menu for the astrological trading framework"""
    while True:
        print("\n=== ASTROLOGICAL TRADING FRAMEWORK ===")
        print("1. Analyze Gold (GC=F)")
        print("2. Analyze S&P 500 (ES=F)")
        print("3. Analyze EUR/USD")
        print("4. Analyze Bitcoin (BTC/USD)")
        print("5. Analyze Custom Symbol")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            analyze_gold()
        elif choice == '2':
            analyze_sp500()
        elif choice == '3':
            analyze_eurusd()
        elif choice == '4':
            analyze_bitcoin()
        elif choice == '5':
            analyze_custom_symbol()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
