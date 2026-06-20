# ============================================================
#  CodeAlpha Data Science Internship — Task 2
#  Unemployment Analysis with Python
#  Author : Chandra Hasini
#  GitHub  : CodeAlpha_UnemploymentAnalysis
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 — ANALYZE UNEMPLOYMENT RATE DATA
# ============================================================
df = pd.read_csv('Unemployment_in_India.csv')
df.columns = df.columns.str.strip()
df.rename(columns={
    'Estimated Unemployment Rate (%)':        'Unemployment_Rate',
    'Estimated Employed':                     'Employed',
    'Estimated Labour Participation Rate (%)':'Labour_Participation_Rate'
}, inplace=True)

print("STEP 1 — Dataset Overview")
print(f"Shape      : {df.shape}")
print(f"Regions    : {df['Region'].nunique()}")
print(f"Area Types : {df['Area'].dropna().unique()}")
print(df.head())

# ============================================================
# STEP 2 — DATA CLEANING, EXPLORATION & VISUALIZATION
# ============================================================
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')
df = df.dropna(subset=['Unemployment_Rate']).sort_values('Date').reset_index(drop=True)
df['Month_Year'] = df['Date'].dt.to_period('M')

print("\nSTEP 2 — After Cleaning")
print(f"Rows remaining : {df.shape[0]}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Date range     : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(df[['Unemployment_Rate','Employed','Labour_Participation_Rate']].describe().round(2))

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Step 2 — Data Cleaning, Exploration & Visualization',
             fontsize=16, fontweight='bold', y=1.01)

# Plot 1 — Distribution of Unemployment Rate
ax = axes[0][0]
ax.hist(df['Unemployment_Rate'], bins=30, color='blue', edgecolor='white', alpha=0.85)
ax.axvline(df['Unemployment_Rate'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean: {df['Unemployment_Rate'].mean():.1f}%")
ax.set_title('Distribution of Unemployment Rate', fontsize=12, fontweight='bold')
ax.set_xlabel('Unemployment Rate (%)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')

# Plot 2 — Rural vs Urban Boxplot
ax = axes[0][1]
data_rural = df[df['Area']=='Rural']['Unemployment_Rate'].dropna()
data_urban = df[df['Area']=='Urban']['Unemployment_Rate'].dropna()
bp = ax.boxplot([data_rural, data_urban], labels=['Rural','Urban'],
                patch_artist=True, widths=0.4,
                boxprops=dict(facecolor='blue', color='black', alpha=0.7),
                medianprops=dict(color='red', linewidth=2.5))
bp['boxes'][1].set_facecolor('green')
ax.set_title('Unemployment Rate — Rural vs Urban', fontsize=12, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Plot 3 — Monthly Trend
ax = axes[1][0]
monthly = df.groupby('Month_Year')['Unemployment_Rate'].mean().reset_index()
monthly['Date'] = monthly['Month_Year'].dt.to_timestamp()
ax.fill_between(monthly['Date'], monthly['Unemployment_Rate'], alpha=0.15, color='blue')
ax.plot(monthly['Date'], monthly['Unemployment_Rate'],
        color='blue', linewidth=2.5, marker='o', markersize=6)
ax.set_title('Monthly Unemployment Rate Trend', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg Unemployment Rate (%)')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=30)

# Plot 4 — Labour Participation Rate
ax = axes[1][1]
lpr = df.groupby(['Month_Year','Area'])['Labour_Participation_Rate'].mean().reset_index()
lpr['Date'] = lpr['Month_Year'].dt.to_timestamp()
for area, color in zip(['Rural','Urban'], ['green','blue']):
    sub = lpr[lpr['Area']==area]
    ax.plot(sub['Date'], sub['Labour_Participation_Rate'],
            label=area, color=color, linewidth=2.2, marker='s', markersize=5)
ax.set_title('Labour Participation Rate — Rural vs Urban', fontsize=12, fontweight='bold')
ax.set_ylabel('Labour Participation Rate (%)')
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('step2_cleaning_exploration.png', dpi=160, bbox_inches='tight')
print("\nStep 2 chart saved!")

# ============================================================
# STEP 3 — COVID-19 IMPACT ON UNEMPLOYMENT
# ============================================================
df['Period'] = df['Date'].apply(
    lambda x: 'During COVID' if x >= pd.Timestamp('2020-03-01') else 'Pre COVID'
)
covid = df.groupby('Period')['Unemployment_Rate'].mean().round(2)
print(f"\nSTEP 3 — COVID Impact")
print(f"Pre COVID avg    : {covid['Pre COVID']}%")
print(f"During COVID avg : {covid['During COVID']}%")
print(f"Increase         : +{covid['During COVID'] - covid['Pre COVID']:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Step 3 — COVID-19 Impact on Unemployment',
             fontsize=16, fontweight='bold')

# Plot 1 — Trend with COVID marker
ax = axes[0]
ax.fill_between(monthly['Date'], monthly['Unemployment_Rate'], alpha=0.15, color='blue')
ax.plot(monthly['Date'], monthly['Unemployment_Rate'],
        color='blue', linewidth=2.5, marker='o', markersize=7)
ax.axvline(pd.Timestamp('2020-03-25'), color='red', linestyle='--', linewidth=2.2)
ax.axvspan(pd.Timestamp('2020-03-01'), monthly['Date'].max(),
           alpha=0.08, color='red', label='COVID Period')
ax.text(pd.Timestamp('2020-03-28'), monthly['Unemployment_Rate'].max()*0.88,
        'COVID-19\nLockdown →', color='red', fontsize=10, fontweight='bold')
ax.set_title('Unemployment Trend — Before & After COVID', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Unemployment Rate (%)')
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=30)

# Plot 2 — Pre vs During COVID Bar Chart
ax = axes[1]
period_area = df.groupby(['Period','Area'])['Unemployment_Rate'].mean().reset_index()
periods = ['Pre COVID', 'During COVID']
x = range(len(periods))
width = 0.35
for i, (area, color) in enumerate(zip(['Rural','Urban'], ['green','blue'])):
    vals = [period_area[(period_area['Period']==p) & (period_area['Area']==area)]['Unemployment_Rate'].values
            for p in periods]
    vals = [v[0] if len(v)>0 else 0 for v in vals]
    offset = (i - 0.5) * width
    rects = ax.bar([xi+offset for xi in x], vals, width,
                   label=area, color=color, alpha=0.85, edgecolor='white')
    for rect, val in zip(rects, vals):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.4,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(list(x))
ax.set_xticklabels(periods, fontsize=11)
ax.set_title('Pre-COVID vs During COVID\n(Rural & Urban)', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Unemployment Rate (%)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('step3_covid_impact.png', dpi=160, bbox_inches='tight')
print("Step 3 chart saved!")

# ============================================================
# STEP 4 — KEY PATTERNS & SEASONAL TRENDS
# ============================================================
monthly_avg = df.groupby('Month_Year')['Unemployment_Rate'].mean()
print(f"\nSTEP 4 — Patterns & Trends")
print(f"Peak Month   : {monthly_avg.idxmax()} ({monthly_avg.max():.2f}%)")
print(f"Lowest Month : {monthly_avg.idxmin()} ({monthly_avg.min():.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Step 4 — Key Patterns & Seasonal Trends',
             fontsize=16, fontweight='bold')

# Plot 1 — Top 10 States
ax = axes[0]
state_avg = df.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False).head(10)
simple_colors = ['blue','orange','green','red','purple',
                 'brown','pink','gray','olive','cyan']
bars = ax.barh(state_avg.index[::-1], state_avg.values[::-1],
               color=simple_colors, edgecolor='white')
for bar, val in zip(bars, state_avg.values[::-1]):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, color='black', fontweight='bold')
ax.set_title('Top 10 States — Highest Avg Unemployment', fontsize=13, fontweight='bold')
ax.set_xlabel('Avg Unemployment Rate (%)')
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

# Plot 2 — Region-wise Heatmap
ax = axes[1]
pivot = df.groupby(['Region','Month_Year'])['Unemployment_Rate'].mean().unstack()
pivot.columns = [str(c) for c in pivot.columns]
pivot = pivot.fillna(pivot.mean())
sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', linewidths=0.3,
            linecolor='white', cbar_kws={'label':'Unemployment Rate (%)','shrink':0.7})
ax.set_title('Region-wise Monthly Heatmap\n(Red = High, Green = Low)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Region')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)

plt.tight_layout()
plt.savefig('step4_patterns_trends.png', dpi=160, bbox_inches='tight')
print("Step 4 chart saved!")

# ============================================================
# STEP 5 — POLICY INSIGHTS
# ============================================================
national_avg = df['Unemployment_Rate'].mean()
state_all    = df.groupby('Region')['Unemployment_Rate'].mean().sort_values()
peak_month   = monthly_avg.idxmax()
peak_val     = monthly_avg.max()

print(f"\nSTEP 5 — Policy Insights")
print(f"National Avg         : {national_avg:.2f}%")
print(f"Most Affected State  : {state_all.index[-1]} ({state_all.values[-1]:.2f}%)")
print(f"Least Affected State : {state_all.index[0]} ({state_all.values[0]:.2f}%)")
print(f"Urban avg            : {df[df['Area']=='Urban']['Unemployment_Rate'].mean():.2f}%")
print(f"Rural avg            : {df[df['Area']=='Rural']['Unemployment_Rate'].mean():.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Step 5 — Insights for Economic & Social Policy',
             fontsize=16, fontweight='bold')

# Plot 1 — Best vs Worst States
ax = axes[0]
combined   = pd.concat([state_all.head(5), state_all.tail(5)])
bar_colors = ['green']*5 + ['red']*5
bars = ax.barh(combined.index, combined.values,
               color=bar_colors, edgecolor='white', alpha=0.85)
for bar, val in zip(bars, combined.values):
    ax.text(val+0.2, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
ax.axvline(national_avg, color='black', linestyle='--', linewidth=1.8,
           label=f'National Avg: {national_avg:.1f}%')
ax.set_title('Best vs Worst Performing States\n(Policy Priority)', fontsize=12, fontweight='bold')
ax.set_xlabel('Avg Unemployment Rate (%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

# Plot 2 — Recovery Trend
ax = axes[1]
colors_line = ['green' if d < pd.Timestamp('2020-03-01') else 'red'
               for d in monthly['Date']]
ax.fill_between(monthly['Date'], monthly['Unemployment_Rate'], alpha=0.12, color='blue')
for i in range(len(monthly)-1):
    ax.plot(monthly['Date'].iloc[i:i+2], monthly['Unemployment_Rate'].iloc[i:i+2],
            color=colors_line[i], linewidth=2.5)
ax.scatter(monthly['Date'], monthly['Unemployment_Rate'],
           color=colors_line, s=60, zorder=5)
ax.axvline(pd.Timestamp('2020-03-25'), color='red', linestyle='--', linewidth=2)
ax.annotate(f'Peak: {peak_val:.1f}%\n({peak_month})',
            xy=(pd.Timestamp('2020-05-31'), peak_val),
            xytext=(pd.Timestamp('2020-01-01'), peak_val*0.92),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=9, fontweight='bold', color='red')
ax.set_title('Unemployment Trend\nGreen = Pre COVID   Red = During COVID',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Avg Unemployment Rate (%)')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('step5_policy_insights.png', dpi=160, bbox_inches='tight')
print("Step 5 chart saved!")

print("\n✅ ALL 5 STEPS COMPLETE — Task 2 Done!")
