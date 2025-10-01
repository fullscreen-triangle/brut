"""
Total Daily Energy Expenditure (TDEE) - complete daily caloric burn
Active Energy Expenditure - calories from intentional activity
Basal Metabolic Rate (BMR) - resting metabolic rate
Thermic Effect of Food - energy cost of digestion
Activity Thermogenesis - heat production from movement
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

def total_daily_energy_expenditure(activity_data: Dict[str, Any], user_profile: Dict[str, Any] = None) -> float:
    """Calculate complete daily caloric burn (TDEE)"""
    # Extract calories from activity data if available
    if 'calories' in activity_data:
        return float(activity_data['calories'])
    
    # Otherwise estimate from activity and user profile
    bmr_val = basal_metabolic_rate(user_profile if user_profile else {})
    active_val = active_energy_expenditure(activity_data)
    tef_val = thermic_effect_of_food(activity_data.get('calories_consumed', 2000))
    neat_val = activity_thermogenesis(activity_data)
    
    tdee = bmr_val + active_val + tef_val + neat_val
    return float(tdee)

def active_energy_expenditure(activity_data: Dict[str, Any]) -> float:
    """Calculate calories from intentional activity"""
    # Extract active calories if available
    active_calories = activity_data.get('active_calories', 0.0)
    if active_calories > 0:
        return float(active_calories)
    
    # Estimate from activity data
    steps = activity_data.get('steps', 0)
    active_minutes = activity_data.get('active_minutes', 0.0)
    
    # Rough estimation: 0.04 calories per step + 5 calories per active minute
    estimated_active = (steps * 0.04) + (active_minutes * 5)
    return float(estimated_active)

def basal_metabolic_rate(user_profile: Dict[str, Any]) -> float:
    """Calculate resting metabolic rate using Mifflin-St Jeor equation"""
    # Default values if user profile not provided
    weight_kg = user_profile.get('weight_kg', 70.0)  # Default 70kg
    height_cm = user_profile.get('height_cm', 170.0)  # Default 170cm
    age_years = user_profile.get('age_years', 30)     # Default 30 years
    sex = user_profile.get('sex', 'male')             # Default male
    
    # Mifflin-St Jeor Equation
    if sex.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) + 5
    else:  # female
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) - 161
    
    return float(bmr)

def thermic_effect_of_food(calories_consumed: float) -> float:
    """Calculate energy cost of digestion (typically 8-10% of total intake)"""
    tef_percentage = 0.09  # 9% of total calories
    tef = calories_consumed * tef_percentage
    return float(tef)

def activity_thermogenesis(activity_data: Dict[str, Any]) -> float:
    """Calculate heat production from movement (NEAT - Non-Exercise Activity Thermogenesis)"""
    # Extract relevant activity indicators
    steps = activity_data.get('steps', 0)
    sedentary_minutes = activity_data.get('sedentary_minutes', 0.0)
    
    # Base NEAT (varies greatly between individuals, 150-350 calories typical)
    base_neat = 200.0  # Base value
    
    # Adjust for activity level
    # More steps = higher NEAT, more sedentary time = lower NEAT
    step_factor = min(steps / 10000, 2.0)  # Normalize to 10k steps, cap at 2x
    sedentary_factor = max(0.5, 1.0 - (sedentary_minutes / 1440))  # Reduce for excessive sitting
    
    neat = base_neat * step_factor * sedentary_factor
    return float(neat)

def analyze_energy_expenditure(activity_record: Dict[str, Any], user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze complete energy expenditure metrics"""
    
    # Calculate all energy expenditure components
    bmr_val = basal_metabolic_rate(user_profile if user_profile else {})
    active_val = active_energy_expenditure(activity_record)
    tef_val = thermic_effect_of_food(activity_record.get('calories_consumed', 2000))
    neat_val = activity_thermogenesis(activity_record)
    tdee_val = total_daily_energy_expenditure(activity_record, user_profile)
    
    results = {
        'period_id': activity_record.get('period_id', 0),
        'timestamp': activity_record.get('timestamp', 0),
        'total_daily_energy_expenditure': tdee_val,
        'active_energy_expenditure': active_val,
        'basal_metabolic_rate': bmr_val,
        'thermic_effect_of_food': tef_val,
        'activity_thermogenesis_neat': neat_val,
        'bmr_percentage': (bmr_val / tdee_val * 100) if tdee_val > 0 else 0,
        'active_percentage': (active_val / tdee_val * 100) if tdee_val > 0 else 0,
        'tef_percentage': (tef_val / tdee_val * 100) if tdee_val > 0 else 0,
        'neat_percentage': (neat_val / tdee_val * 100) if tdee_val > 0 else 0
    }
    
    # Add energy balance if calories consumed available
    calories_consumed = activity_record.get('calories_consumed', None)
    if calories_consumed is not None:
        results['calories_consumed'] = float(calories_consumed)
        results['energy_balance'] = float(calories_consumed - tdee_val)
        results['energy_balance_category'] = 'surplus' if results['energy_balance'] > 0 else 'deficit'
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of energy expenditure metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    energy_data = []
    for result in results:
        energy_data.append({
            'period_id': result['period_id'],
            'tdee': result['total_daily_energy_expenditure'],
            'active_calories': result['active_energy_expenditure'],
            'bmr': result['basal_metabolic_rate'],
            'tef': result['thermic_effect_of_food'],
            'neat': result['activity_thermogenesis_neat'],
            'bmr_pct': result['bmr_percentage'],
            'active_pct': result['active_percentage'],
            'tef_pct': result['tef_percentage'],
            'neat_pct': result['neat_percentage']
        })
    
    df = pd.DataFrame(energy_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # TDEE over time
    axes[0,0].plot(df['period_id'], df['tdee'], 'o-', alpha=0.7, color='red')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('TDEE (calories)')
    axes[0,0].set_title('Total Daily Energy Expenditure')
    axes[0,0].grid(True, alpha=0.3)
    
    # Energy expenditure components
    width = 0.8
    bottom_bmr = df['bmr']
    bottom_active = bottom_bmr + df['active_calories']
    bottom_tef = bottom_active + df['tef']
    
    axes[0,1].bar(df['period_id'], df['bmr'], width, label='BMR', alpha=0.8)
    axes[0,1].bar(df['period_id'], df['active_calories'], width, bottom=bottom_bmr, label='Active', alpha=0.8)
    axes[0,1].bar(df['period_id'], df['tef'], width, bottom=bottom_active, label='TEF', alpha=0.8)
    axes[0,1].bar(df['period_id'], df['neat'], width, bottom=bottom_tef, label='NEAT', alpha=0.8)
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('Calories')
    axes[0,1].set_title('Energy Expenditure Components (Stacked)')
    axes[0,1].legend()
    
    # Energy expenditure percentages (pie chart of averages)
    avg_percentages = [df['bmr_pct'].mean(), df['active_pct'].mean(), 
                      df['tef_pct'].mean(), df['neat_pct'].mean()]
    labels = ['BMR', 'Active', 'TEF', 'NEAT']
    axes[1,0].pie(avg_percentages, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Average Energy Expenditure Distribution')
    
    # Active calories vs BMR
    axes[1,1].scatter(df['bmr'], df['active_calories'], alpha=0.7)
    axes[1,1].set_xlabel('BMR (calories)')
    axes[1,1].set_ylabel('Active Calories')
    axes[1,1].set_title('Active Calories vs BMR')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_expenditure_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional detailed breakdown
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # BMR vs TDEE
    axes[0,0].scatter(df['bmr'], df['tdee'], alpha=0.7)
    axes[0,0].plot([df['bmr'].min(), df['bmr'].max()], 
                   [df['bmr'].min(), df['bmr'].max()], 'r--', alpha=0.5, label='BMR = TDEE')
    axes[0,0].set_xlabel('BMR (calories)')
    axes[0,0].set_ylabel('TDEE (calories)')
    axes[0,0].set_title('BMR vs TDEE')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Component trends
    axes[0,1].plot(df['period_id'], df['bmr'], 'o-', alpha=0.7, label='BMR')
    axes[0,1].plot(df['period_id'], df['active_calories'], 's-', alpha=0.7, label='Active')
    axes[0,1].plot(df['period_id'], df['neat'], '^-', alpha=0.7, label='NEAT')
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('Calories')
    axes[0,1].set_title('Energy Component Trends')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Distribution of components
    component_data = [df['bmr'], df['active_calories'], df['tef'], df['neat']]
    axes[1,0].boxplot(component_data, labels=['BMR', 'Active', 'TEF', 'NEAT'])
    axes[1,0].set_ylabel('Calories')
    axes[1,0].set_title('Energy Component Distributions')
    axes[1,0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_cols = ['tdee', 'bmr', 'active_calories', 'neat', 'tef']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Energy Metrics Correlations')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_expenditure_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Energy expenditure visualizations saved to {output_dir}/")

def main():
    """Main function to analyze energy expenditure metrics"""
    
    print("Energy Expenditure Metrics Analysis")
    print("=" * 50)
    
    # Load activity data - using sleep data and creating mock activity data
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} records for energy analysis")
    
    # Mock user profile
    user_profile = {
        'weight_kg': 75.0,
        'height_cm': 175.0,
        'age_years': 35,
        'sex': 'male'
    }
    
    # Analyze first 10 records
    results = []
    for i, record in enumerate(sleep_data[:10]):
        print(f"Analyzing record {i+1}/10...")
        
        # Create mock activity data from sleep record
        activity_record = {
            'period_id': record.get('period_id', i),
            'timestamp': record.get('bedtime_start_dt_adjusted', 0),
            'steps': np.random.randint(3000, 12000),  # Mock step data
            'active_minutes': np.random.uniform(20, 90),  # Mock active minutes
            'sedentary_minutes': np.random.uniform(400, 800),  # Mock sedentary time
            'calories_consumed': np.random.uniform(1800, 2500),  # Mock food intake
            'active_calories': np.random.uniform(200, 600)  # Mock active calories
        }
        
        result = analyze_energy_expenditure(activity_record, user_profile)
        results.append(result)
    
    # Save results
    output_dir = '../results/energy_expenditure'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/energy_expenditure_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/energy_expenditure_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print summary statistics
    print("\nEnergy Expenditure Summary:")
    print("-" * 40)
    
    tdee_vals = [r['total_daily_energy_expenditure'] for r in results]
    active_vals = [r['active_energy_expenditure'] for r in results]
    bmr_vals = [r['basal_metabolic_rate'] for r in results]
    
    print(f"TDEE - Mean: {np.mean(tdee_vals):.0f} cal, Range: {np.min(tdee_vals):.0f}-{np.max(tdee_vals):.0f} cal")
    print(f"Active Calories - Mean: {np.mean(active_vals):.0f} cal, Range: {np.min(active_vals):.0f}-{np.max(active_vals):.0f} cal")
    print(f"BMR - Mean: {np.mean(bmr_vals):.0f} cal (consistent as same user profile)")
    
    # Energy balance summary
    energy_balances = [r.get('energy_balance', 0) for r in results if 'energy_balance' in r]
    if energy_balances:
        surplus_days = sum(1 for eb in energy_balances if eb > 0)
        deficit_days = len(energy_balances) - surplus_days
        print(f"Energy Balance - Surplus days: {surplus_days}, Deficit days: {deficit_days}")
        print(f"Average energy balance: {np.mean(energy_balances):.0f} cal/day")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()