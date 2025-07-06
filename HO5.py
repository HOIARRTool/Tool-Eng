# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================
import streamlit as st
import io
import urllib.parse
import json
import re
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import google.generativeai as genai

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
LOGO_URL = "https://raw.githubusercontent.com/HOIARRTool/hoiarr/refs/heads/main/logo1.png"
st.set_page_config(page_title="HOIA-RR", page_icon=LOGO_URL, layout="wide")
st.markdown("""
<style>
/* CSS to style the text area inside the chat input */
[data-testid="stChatInput"] textarea {
    min-height: 80px;
    height: 100px;
    resize: vertical;
    background-color: transparent;
    border: none;
}
</style>
""", unsafe_allow_html=True)
# --- START: CSS Styles ---
st.markdown("""
<style>
    /* CSS Styles for Print View Only */
    @media print {
        /* === FIX FOR st.columns TO STAY IN A ROW === */
        div[data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: repeat(5, 1fr) !important;
            gap: 1.2rem !important;
        }
        /* === FIX FOR TABLES & DATAFRAMES SPANNING PAGES === */
        .stDataFrame, .stTable { break-inside: avoid; page-break-inside: avoid; }
        thead, tr, th, td { break-inside: avoid !important; page-break-inside: avoid !important; }
        h1, h2, h3, h4, h5 { page-break-after: avoid; }
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* General CSS for Header */
.custom-header { font-size: 20px; font-weight: bold; margin-top: 0px !important; padding-top: 0px !important; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] {
    border: 1px solid #ddd; padding: 0.75rem; border-radius: 0.5rem; height: 100%;
    display: flex; flex-direction: column; justify-content: center;}
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] { background-color: #e6fffa; border-color: #b2f5ea; }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { background-color: #fff3e0; border-color: #ffe0b2; }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { background-color: #fce4ec; border-color: #f8bbd0; }
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] { background-color: #e3f2fd; border-color: #bbdefb; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div,
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricDelta"]
{ color: #262730 !important; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div { font-size: 0.8rem !important; line-height: 1.2 !important; white-space: normal !important; overflow-wrap: break-word !important; word-break: break-word; display: block !important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
div[data-testid="stHorizontalBlock"] > div .stExpander { border: none !important; box-shadow: none !important; padding: 0 !important; margin-top: 0.5rem;}
div[data-testid="stHorizontalBlock"] > div .stExpander header { padding: 0.25rem 0.5rem !important; font-size: 0.75rem !important; border-radius: 0.25rem;}
div[data-testid="stHorizontalBlock"] > div .stExpander div[data-testid="stExpanderDetails"] { max-height: 200px; overflow-y: auto;}
.stDataFrame table td, .stDataFrame table th { color: black !important; font-size: 0.9rem !important; }
.stDataFrame table th { font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)


# --- END: CSS Styles ---


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================
def image_to_base64(img_path_str):
    img_path = Path(img_path_str)
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None


def extract_severity_level(prompt_text):
    # This function parses input data which might be in Thai or English.
    # The regex '(?:ระดับ|LEVEL)' is intentionally kept to handle both.
    prompt_upper = prompt_text.upper()
    match1 = re.search(r'(?:ระดับ|LEVEL)\s*([A-I])\b', prompt_upper)
    if match1:
        return (match1.group(1), 'clinical')
    match2 = re.search(r'(?:ระดับ|LEVEL)\s*([1-5])\b', prompt_upper)
    if match2:
        return (match2.group(1), 'general')
    match3 = re.search(r'\b([A-I])\b', prompt_upper)
    if match3:
        return (match3.group(1), 'clinical')
    match4 = re.search(r'\b([1-5])\b', prompt_upper)
    if match4:
        return (match4.group(1), 'general')
    return (None, None)


def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl', keep_default_na=False)
        return df
    except Exception as e:
        st.session_state.upload_error_message = f"Error reading the Excel file: {e}"
        return pd.DataFrame()


def calculate_autocorrelation(series: pd.Series):
    if len(series) < 2: return 0.0
    if series.std() == 0: return 1.0
    try:
        series_mean = series.mean()
        c1 = np.sum((series.iloc[1:].values - series_mean) * (series.iloc[:-1].values - series_mean))
        c0 = np.sum((series.values - series_mean) ** 2)
        return c1 / c0 if c0 != 0 else 0.0
    except Exception:
        return 0.0


@st.cache_data
def calculate_risk_level_trend(_df: pd.DataFrame):
    risk_level_map_to_score = {
        "51": 21, "52": 22, "53": 23, "54": 24, "55": 25, "41": 16, "42": 17, "43": 18, "44": 19, "45": 20,
        "31": 11, "32": 12, "33": 13, "34": 14, "35": 15, "21": 6, "22": 7, "23": 8, "24": 9, "25": 10,
        "11": 1, "12": 2, "13": 3, "14": 4, "15": 5
    }
    # These column names are from the source Excel file and should not be translated
    if _df.empty or 'รหัส' not in _df.columns or 'Occurrence Date' not in _df.columns or 'Risk Level' not in _df.columns:
        return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Occurrence Date', 'Risk Level']].copy()
    analysis_df['Ordinal_Risk_Score'] = analysis_df['Risk Level'].astype(str).map(risk_level_map_to_score)
    analysis_df.dropna(subset=['Ordinal_Risk_Score'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    analysis_df['YearMonth'] = pd.to_datetime(analysis_df['Occurrence Date'], errors='coerce').dt.to_period('M')
    results = []
    all_incident_codes = analysis_df['รหัส'].unique()
    for code in all_incident_codes:
        incident_subset = analysis_df[analysis_df['รหัส'] == code]
        monthly_risk_score = incident_subset.groupby('YearMonth')['Ordinal_Risk_Score'].mean().reset_index()
        slope = 0
        if len(monthly_risk_score) >= 3:
            X = np.arange(len(monthly_risk_score)).reshape(-1, 1)
            y = monthly_risk_score['Ordinal_Risk_Score']
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
        average_risk_score = incident_subset['Ordinal_Risk_Score'].mean()
        results.append({
            'รหัส': code,
            'Average_Risk_Score': average_risk_score,
            'Data_Points_Months': len(monthly_risk_score),
            'Risk_Level_Trend_Slope': slope
        })
    if not results: return pd.DataFrame()
    final_df = pd.DataFrame(results)
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(final_df, incident_names, on='รหัส', how='left')
    final_df = final_df.sort_values(by='Risk_Level_Trend_Slope', ascending=False)
    return final_df


@st.cache_data
def calculate_persistence_risk_score(_df: pd.DataFrame, total_months: int):
    risk_level_map_to_score = {
        "51": 21, "52": 22, "53": 23, "54": 24, "55": 25, "41": 16, "42": 17, "43": 18, "44": 19, "45": 20,
        "31": 11, "32": 12, "33": 13, "34": 14, "35": 15, "21": 6, "22": 7, "23": 8, "24": 9, "25": 10,
        "11": 1, "12": 2, "13": 3, "14": 4, "15": 5
    }
    if _df.empty or 'รหัส' not in _df.columns or 'Risk Level' not in _df.columns:
        return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Risk Level']].copy()
    analysis_df['Ordinal_Risk_Score'] = analysis_df['Risk Level'].astype(str).map(risk_level_map_to_score)
    analysis_df.dropna(subset=['Ordinal_Risk_Score'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    persistence_metrics = analysis_df.groupby('รหัส').agg(
        Average_Ordinal_Risk_Score=('Ordinal_Risk_Score', 'mean'),
        Total_Occurrences=('รหัส', 'size')
    ).reset_index()
    if total_months == 0: total_months = 1
    persistence_metrics['Incident_Rate_Per_Month'] = persistence_metrics['Total_Occurrences'] / total_months
    max_rate = persistence_metrics['Incident_Rate_Per_Month'].max()
    if max_rate == 0: max_rate = 1
    persistence_metrics['Frequency_Score'] = persistence_metrics['Incident_Rate_Per_Month'] / max_rate
    persistence_metrics['Avg_Severity_Score'] = persistence_metrics['Average_Ordinal_Risk_Score'] / 25.0
    persistence_metrics['Persistence_Risk_Score'] = persistence_metrics['Frequency_Score'] + persistence_metrics[
        'Avg_Severity_Score']
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(persistence_metrics, incident_names, on='รหัส', how='left')
    final_df = final_df.sort_values(by='Persistence_Risk_Score', ascending=False)
    return final_df


@st.cache_data
def calculate_frequency_trend_poisson(_df: pd.DataFrame):
    if _df.empty or 'รหัส' not in _df.columns or 'Occurrence Date' not in _df.columns:
        return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Occurrence Date']].copy()
    analysis_df.dropna(subset=['Occurrence Date'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    analysis_df['YearMonth'] = pd.to_datetime(analysis_df['Occurrence Date']).dt.to_period('M')
    full_date_range = pd.period_range(start=analysis_df['YearMonth'].min(), end=analysis_df['YearMonth'].max(),
                                      freq='M')
    results = []
    all_incident_codes = analysis_df['รหัส'].unique()
    for code in all_incident_codes:
        incident_subset = analysis_df[analysis_df['รหัส'] == code]
        MIN_OCCURRENCES = 3
        if len(incident_subset) < MIN_OCCURRENCES:
            continue
        monthly_counts = incident_subset.groupby('YearMonth').size().reindex(full_date_range, fill_value=0)
        if len(monthly_counts) < 2:
            continue
        y = monthly_counts.values
        X = np.arange(len(monthly_counts))
        X = sm.add_constant(X)
        try:
            model = sm.Poisson(y, X).fit(disp=0)
            time_coefficient = model.params[1]
            results.append({
                'รหัส': code,
                'Poisson_Trend_Slope': time_coefficient,
                'Total_Occurrences': y.sum(),
                'Months_Observed': len(y)
            })
        except Exception:
            continue
    if not results: return pd.DataFrame()
    final_df = pd.DataFrame(results)
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(final_df, incident_names, on='รหัส', how='left')
    final_df = final_df.sort_values(by='Poisson_Trend_Slope', ascending=False)
    return final_df


def create_poisson_trend_plot(df, selected_code_for_plot, display_df):
    full_date_range_for_plot = pd.period_range(
        start=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').min(),
        end=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').max(),
        freq='M'
    )
    subset_for_plot = df[df['รหัส'] == selected_code_for_plot].copy()
    subset_for_plot['YearMonth'] = pd.to_datetime(subset_for_plot['Occurrence Date']).dt.to_period('M')
    counts_for_plot = subset_for_plot.groupby('YearMonth').size().reindex(full_date_range_for_plot, fill_value=0)
    y_plot = counts_for_plot.values
    trend_line_for_plot = []
    try:
        X_plot_raw = np.arange(len(counts_for_plot))
        X_lin_reg = X_plot_raw.reshape(-1, 1)
        lin_reg_model = LinearRegression().fit(X_lin_reg, y_plot)
        trend_line_for_plot = lin_reg_model.predict(X_lin_reg)
    except Exception as e:
        st.error(f"Error calculating linear trend line: {e}")

    fig_plot = go.Figure()
    fig_plot.add_trace(go.Bar(
        x=counts_for_plot.index.strftime('%Y-%m'),
        y=y_plot,
        name='Actual Occurrences',
        marker=dict(color='#AED6F1', cornerradius=8)
    ))
    if len(trend_line_for_plot) > 0:
        fig_plot.add_trace(go.Scatter(
            x=counts_for_plot.index.strftime('%Y-%m'),
            y=trend_line_for_plot,
            mode='lines',
            name='Trend Line (Linear)',
            line=dict(color='red', width=2, dash='dash')
        ))
    fig_plot.update_layout(
        title=f'Incident Distribution: {selected_code_for_plot}',
        xaxis_title='Month-Year',
        yaxis_title='Number of Occurrences',
        barmode='overlay',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)')
    )
    slope_val_for_annot = \
        display_df.loc[display_df['Code'] == selected_code_for_plot, 'Frequency Trend (Slope)'].iloc[0]
    factor_val_for_annot = \
        display_df.loc[display_df['Code'] == selected_code_for_plot, 'Change Rate (factor/month)'].iloc[0]
    annot_text = (f"<b>Poisson Slope: {slope_val_for_annot:.4f}</b><br>"
                  f"Change Rate: x{factor_val_for_annot:.2f} per month")
    fig_plot.add_annotation(
        x=0.5, y=0.98,
        xref="paper", yref="paper",
        text=annot_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(255, 255, 224, 0.7)"
    )
    return fig_plot


def create_goal_summary_table(data_df_goal, goal_category_name_param,
                              e_up_non_numeric_levels_param, e_up_numeric_levels_param=None,
                              is_org_safety_table=False):
    goal_category_name_param = str(goal_category_name_param).strip()
    if 'หมวด' not in data_df_goal.columns:
        return pd.DataFrame()
    df_filtered_by_goal_cat = data_df_goal[
        data_df_goal['หมวด'].astype(str).str.strip() == goal_category_name_param].copy()
    if df_filtered_by_goal_cat.empty: return pd.DataFrame()
    if 'Incident Type' not in df_filtered_by_goal_cat.columns or 'Impact' not in df_filtered_by_goal_cat.columns: return pd.DataFrame()
    try:
        pvt_table_goal = pd.crosstab(df_filtered_by_goal_cat['Incident Type'],
                                     df_filtered_by_goal_cat['Impact'].astype(str).str.strip(), margins=True,
                                     margins_name='Total')
    except Exception:
        return pd.DataFrame()
    if 'Total' in pvt_table_goal.index: pvt_table_goal = pvt_table_goal.drop(index='Total')
    if pvt_table_goal.empty: return pd.DataFrame()
    if 'Total' not in pvt_table_goal.columns: pvt_table_goal['Total'] = pvt_table_goal.sum(axis=1)
    all_impact_columns_goal = [str(col).strip() for col in pvt_table_goal.columns if col != 'Total']
    e_up_non_numeric_levels_param_stripped = [str(level).strip() for level in e_up_non_numeric_levels_param]
    e_up_numeric_levels_param_stripped = [str(level).strip() for level in
                                          e_up_numeric_levels_param] if e_up_numeric_levels_param else []
    e_up_columns_goal = [col for col in all_impact_columns_goal if
                         col not in e_up_non_numeric_levels_param_stripped and (
                                 not e_up_numeric_levels_param_stripped or col not in e_up_numeric_levels_param_stripped)]
    report_data_goal = []
    for incident_type_goal, row_data_goal in pvt_table_goal.iterrows():
        total_e_up_count_goal = sum(row_data_goal[col] for col in e_up_columns_goal if
                                    col in row_data_goal.index and pd.notna(row_data_goal[col]))
        total_all_impacts_goal = row_data_goal['Total'] if 'Total' in row_data_goal and pd.notna(
            row_data_goal['Total']) else 0
        percent_e_up_goal = (total_e_up_count_goal / total_all_impacts_goal * 100) if total_all_impacts_goal > 0 else 0
        report_data_goal.append(
            {'Incident Type': incident_type_goal, 'Total E-up': total_e_up_count_goal, '% E-up': percent_e_up_goal})
    report_df_goal = pd.DataFrame(report_data_goal)
    if report_df_goal.empty:
        merged_report_table_goal = pvt_table_goal.reset_index()
        merged_report_table_goal['Total E-up'] = 0
        merged_report_table_goal['% E-up'] = 0.0
    else:
        merged_report_table_goal = pd.merge(pvt_table_goal.reset_index(), report_df_goal, on='Incident Type',
                                            how='outer')
    if 'Total E-up' not in merged_report_table_goal.columns:
        merged_report_table_goal['Total E-up'] = 0
    else:
        merged_report_table_goal['Total E-up'].fillna(0, inplace=True)
    if '% E-up' not in merged_report_table_goal.columns:
        merged_report_table_goal['% E-up'] = 0.0
    else:
        merged_report_table_goal['% E-up'].fillna(0.0, inplace=True)
    cols_to_drop_from_display_goal = [col for col in e_up_non_numeric_levels_param_stripped if
                                      col in merged_report_table_goal.columns]
    if e_up_numeric_levels_param_stripped: cols_to_drop_from_display_goal.extend(
        [col for col in e_up_numeric_levels_param_stripped if col in merged_report_table_goal.columns])
    merged_report_table_goal = merged_report_table_goal.drop(columns=cols_to_drop_from_display_goal, errors='ignore')
    total_col_original_name, e_up_col_name, percent_e_up_col_name = 'Total', 'Total E-up', '% E-up'
    if is_org_safety_table:
        total_col_display_name, e_up_col_display_name, percent_e_up_display_name = 'Total 1-5', 'Total 3-5', '% 3-5'
        merged_report_table_goal.rename(
            columns={total_col_original_name: total_col_display_name, e_up_col_name: e_up_col_display_name,
                     percent_e_up_col_name: percent_e_up_display_name}, inplace=True, errors='ignore')
    else:
        total_col_display_name, e_up_col_display_name, percent_e_up_display_name = 'Total A-I', e_up_col_name, percent_e_up_col_name
        merged_report_table_goal.rename(columns={total_col_original_name: total_col_display_name}, inplace=True,
                                        errors='ignore')
    merged_report_table_goal['Incident Type Name'] = merged_report_table_goal['Incident Type'].map(type_name).fillna(
        merged_report_table_goal['Incident Type'])
    final_columns_goal_order = ['Incident Type Name'] + [col for col in e_up_columns_goal if
                                                         col in merged_report_table_goal.columns] + [
                                   e_up_col_display_name, total_col_display_name, percent_e_up_display_name]
    final_columns_present_goal = [col for col in final_columns_goal_order if col in merged_report_table_goal.columns]
    merged_report_table_goal = merged_report_table_goal[final_columns_present_goal]
    if percent_e_up_display_name in merged_report_table_goal.columns and pd.api.types.is_numeric_dtype(
            merged_report_table_goal[percent_e_up_display_name]):
        try:
            merged_report_table_goal[percent_e_up_display_name] = merged_report_table_goal[
                percent_e_up_display_name].astype(float).map('{:.2f}%'.format)
        except ValueError:
            pass
    return merged_report_table_goal.set_index('Incident Type Name')


def create_severity_table(input_df, row_column_name, table_title, specific_row_order=None):
    if not isinstance(input_df,
                      pd.DataFrame) or input_df.empty or row_column_name not in input_df.columns or 'Impact Level' not in input_df.columns: return None
    temp_df = input_df.copy()
    temp_df['Impact Level'] = temp_df['Impact Level'].astype(str).str.strip().replace('N/A', 'Not Specified')
    if temp_df[row_column_name].dropna().empty: return None
    try:
        severity_crosstab = pd.crosstab(temp_df[row_column_name].astype(str).str.strip(), temp_df['Impact Level'])
    except Exception:
        return None
    impact_level_map_cols = {'1': 'A-B (1)', '2': 'C-D (2)', '3': 'E-F (3)', '4': 'G-H (4)', '5': 'I (5)',
                             'Not Specified': 'Unspecified LV'}
    desired_cols_ordered_keys = ['1', '2', '3', '4', '5', 'Not Specified']
    for col_key in desired_cols_ordered_keys:
        if col_key not in severity_crosstab.columns: severity_crosstab[col_key] = 0
    present_ordered_keys = [key for key in desired_cols_ordered_keys if key in severity_crosstab.columns]
    if not present_ordered_keys: return None
    severity_crosstab = severity_crosstab[present_ordered_keys].rename(columns=impact_level_map_cols)
    final_display_cols_renamed = [impact_level_map_cols[key] for key in present_ordered_keys if
                                  key in impact_level_map_cols]
    if not final_display_cols_renamed: return None
    severity_crosstab['Total All Levels'] = severity_crosstab[
        [col for col in final_display_cols_renamed if col in severity_crosstab.columns]].sum(axis=1)
    if specific_row_order:
        severity_crosstab = severity_crosstab.reindex([str(i) for i in specific_row_order]).fillna(0).astype(int)
    else:
        severity_crosstab = severity_crosstab[severity_crosstab['Total All Levels'] > 0]
    if severity_crosstab.empty: return None
    st.markdown(f"##### {table_title}")
    display_column_order_from_map = [impact_level_map_cols.get(key) for key in desired_cols_ordered_keys]
    display_column_order_present = [col for col in display_column_order_from_map if
                                    col in severity_crosstab.columns] + (
                                       ['Total All Levels'] if 'Total All Levels' in severity_crosstab.columns else [])
    st.dataframe(
        severity_crosstab[[col for col in display_column_order_present if col in severity_crosstab.columns]].astype(
            int), use_container_width=True)
    return severity_crosstab


def create_psg9_summary_table(input_df):
    if not isinstance(input_df,
                      pd.DataFrame) or 'หมวดหมู่มาตรฐานสำคัญ' not in input_df.columns or 'Impact' not in input_df.columns: return None
    psg9_placeholders = ["Not in PSG9 Catalog", "Could not identify (Merge PSG9 failed)",
                         "Could not identify (Check columns in PSG9code.xlsx)",
                         "Could not identify (PSG9code.xlsx not loaded/empty)",
                         "Could not identify (Merge PSG9 failed - rename)",
                         "Could not identify (Merge PSG9 failed - no col)",
                         "Could not identify (PSG9code.xlsx not loaded/incomplete data)"]

    df_filtered = input_df[
        ~input_df['หมวดหมู่มาตรฐานสำคัญ'].isin(psg9_placeholders) & input_df['หมวดหมู่มาตรฐานสำคัญ'].notna()].copy()
    if df_filtered.empty: return pd.DataFrame()
    try:
        summary_table = pd.crosstab(df_filtered['หมวดหมู่มาตรฐานสำคัญ'], df_filtered['Impact'], margins=True,
                                    margins_name='Total A-I')
    except Exception:
        return pd.DataFrame()
    if 'Total A-I' in summary_table.index: summary_table = summary_table.drop(index='Total A-I')
    if summary_table.empty: return pd.DataFrame()
    all_impacts, e_up_impacts = list('ABCDEFGHI'), list('EFGHI')
    for impact_col in all_impacts:
        if impact_col not in summary_table.columns: summary_table[impact_col] = 0
    if 'Total A-I' not in summary_table.columns: summary_table['Total A-I'] = summary_table[
        [col for col in all_impacts if col in summary_table.columns]].sum(axis=1)
    summary_table['Total E-up'] = summary_table[[col for col in e_up_impacts if col in summary_table.columns]].sum(
        axis=1)
    summary_table['% E-up'] = (summary_table['Total E-up'] / summary_table['Total A-I'] * 100).fillna(0)
    psg_order = [PSG9_label_dict[i] for i in sorted(PSG9_label_dict.keys())]
    summary_table = summary_table.reindex(psg_order).fillna(0)
    display_cols_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Total E-up', 'Total A-I', '% E-up']
    final_table = summary_table[[col for col in display_cols_order if col in summary_table.columns]].copy()
    for col in final_table.columns:
        if col != '% E-up': final_table[col] = final_table[col].astype(int)
    final_table['% E-up'] = final_table['% E-up'].map('{:.2f}%'.format)
    return final_table


def get_text_color_for_bg(hex_color):
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6: return '#000000'
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        return '#FFFFFF' if luminance < 0.5 else '#000000'
    except ValueError:
        return '#000000'


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = {}
if 'df_freq_for_display' not in st.session_state:
    st.session_state.df_freq_for_display = pd.DataFrame()
if 'upload_error_message' not in st.session_state:
    st.session_state.upload_error_message = None

analysis_options_list = [
    "Overall Summary Dashboard",
    "Sentinel Events List",
    "Interactive Risk Matrix",
    "Incident Resolution",
    "Incident Summary Graphs",
    "Sankey: Overall View",
    "Sankey: 9 Patient Safety Goals",
    "Scatter Plot & Top 10",
    "Summary by Safety Goals",
    "Analysis by Category & Status",
    "Persistence Risk Index",
    "Frequency Trend (Poisson)",
    "Executive Summary",
    "Chat with AI Assistant",
]

if 'selected_analysis' not in st.session_state:
    st.session_state.selected_analysis = analysis_options_list[0]
if 'show_incident_table' not in st.session_state:
    st.session_state.show_incident_table = False
if 'clicked_risk_impact' not in st.session_state:
    st.session_state.clicked_risk_impact = None
if 'clicked_risk_freq' not in st.session_state:
    st.session_state.clicked_risk_freq = None

# ==============================================================================
# STATIC DATA DEFINITIONS
# ==============================================================================
PSG9_FILE_PATH = "PSG9code.xlsx"
SENTINEL_FILE_PATH = "Sentinel2024.xlsx"
ALLCODE_FILE_PATH = "Code2024.xlsx"
GEMINI_PERSONA_INSTRUCTION = "As an AI assistant, please adopt a helpful and professional tone. Always answer the user's question directly based on the provided context."
psg9_r_codes_for_counting = set()
sentinel_composite_keys = set()
df2 = pd.DataFrame()
PSG9code_df_master = pd.DataFrame()
Sentinel2024_df = pd.DataFrame()
allcode2024_df = pd.DataFrame()

try:
    if Path(PSG9_FILE_PATH).is_file():
        PSG9code_df_master = pd.read_excel(PSG9_FILE_PATH)
        if 'รหัส' in PSG9code_df_master.columns:
            psg9_r_codes_for_counting = set(PSG9code_df_master['รหัส'].astype(str).str.strip().unique())
    if Path(SENTINEL_FILE_PATH).is_file():
        Sentinel2024_df = pd.read_excel(SENTINEL_FILE_PATH)
        if 'รหัส' in Sentinel2024_df.columns and 'Impact' in Sentinel2024_df.columns:
            Sentinel2024_df['รหัส'] = Sentinel2024_df['รหัส'].astype(str).str.strip()
            Sentinel2024_df['Impact'] = Sentinel2024_df['Impact'].astype(str).str.strip()
            Sentinel2024_df.dropna(subset=['รหัส', 'Impact'], inplace=True)
            sentinel_composite_keys = set((Sentinel2024_df['รหัส'] + '-' + Sentinel2024_df['Impact']).unique())
    if Path(ALLCODE_FILE_PATH).is_file():
        allcode2024_df = pd.read_excel(ALLCODE_FILE_PATH)
        if 'รหัส' in allcode2024_df.columns and all(
                col in allcode2024_df.columns for col in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]):
            df2 = allcode2024_df[["รหัส", "ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]].drop_duplicates().copy()
            df2['รหัส'] = df2['รหัส'].astype(str).str.strip()
except FileNotFoundError as e:
    st.session_state.upload_error_message = f"Definition file not found: {e}. Please place the file in the same folder as the program."
except Exception as e:
    st.session_state.upload_error_message = f"Error loading definition file: {e}"

color_discrete_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green', 'Undefined': '#D3D3D3'}
month_label = {1: '01 January', 2: '02 February', 3: '03 March', 4: '04 April', 5: '05 May', 6: '06 June',
               7: '07 July', 8: '08 August', 9: '09 September', 10: '10 October', 11: '11 November', 12: '12 December'}

severity_definitions_clinical = {
    'A': "A (Near Miss - Internal): An event occurred, was discovered internally, and corrected without affecting others, patients, or staff.",
    'B': "B (Near Miss - External): An event/error occurred and was passed on to others but was detected and corrected before any impact on patients or staff.",
    'C': "C (No Harm): An event/error occurred and reached the patient or staff but caused no harm or damage.",
    'D': "D (Monitoring Required): An error occurred, reaching the patient or staff, requiring special monitoring to ensure no harm results.",
    'E': "E (Temporary Harm): An error occurred, causing temporary harm to the patient or staff that required intervention/treatment.",
    'F': "F (Prolonged Harm): An error occurred, causing an impact that required a longer than usual or extended correction period; patient or staff required prolonged treatment/hospitalization.",
    'G': "G (Permanent Harm): An error occurred, causing permanent disability to the patient or staff, or an impact that damaged reputation/credibility and/or led to complaints.",
    'H': "H (Life-Sustaining Intervention): An error occurred, requiring life-saving intervention for the patient or staff, or causing reputational damage and/or claims for damages against the hospital.",
    'I': "I (Patient Death): An error occurred that was a contributing factor to a patient's or staff member's death, or caused severe reputational damage with legal action/media coverage."
}
severity_definitions_general = {
    '1': "Level 1: An error occurred with no impact on the success or objectives of the operation (*financial impact of 0 - 10,000 THB).",
    '2': "Level 2: An error occurred with a controllable impact on the success or objectives of the operation (*financial impact of 10,001 - 50,000 THB).",
    '3': "Level 3: An error occurred with an impact requiring correction on the success or objectives of the operation (*financial impact of 50,001 - 250,000 THB).",
    '4': "Level 4: An error occurred, causing the operation to fail to meet its goals (*financial impact of 250,001 – 10,000,000 THB).",
    '5': "Level 5: An error occurred, causing the operation to fail to meet its goals and severely damaging the organization's mission (*financial impact of over 10 million THB)."
}
PSG9_label_dict = {
    1: '01 Wrong patient, wrong site, wrong procedure',
    2: '02 Healthcare-associated infections in personnel',
    3: '03 Major infections (SSI, VAP, CAUTI, CLABSI)',
    4: '04 Medication Error and Adverse Drug Event',
    5: '05 Blood transfusion error (wrong person, group, type)',
    6: '06 Patient identification error',
    7: '07 Diagnostic error',
    8: '08 Inaccurate laboratory/pathology report',
    9: '09 Triage error in the emergency room'
}
type_name = {'CPS': 'Safe Surgery', 'CPI': 'Infection Prevention and Control', 'CPM': 'Medication & Blood Safety',
             'CPP': 'Patient Care Process', 'CPL': 'Line, Tube & Catheter and Laboratory', 'CPE': 'Emergency Response',
             'CSG': 'Gynecology & Obstetrics diseases and procedure', 'CSS': 'Surgical diseases and procedure',
             'CSM': 'Medical diseases and procedure', 'CSP': 'Pediatric diseases and procedure',
             'CSO': 'Orthopedic diseases and procedure', 'CSD': 'Dental diseases and procedure',
             'GPS': 'Social Media and Communication', 'GPI': 'Infection and Exposure',
             'GPM': 'Mental Health and Mediation', 'GPP': 'Process of work', 'GPL': 'Lane (Traffic) and Legal Issues',
             'GPE': 'Environment and Working Conditions', 'GOS': 'Strategy, Structure, Security',
             'GOI': 'Information Technology & Communication, Internal control & Inventory',
             'GOM': 'Manpower, Management', 'GOP': 'Policy, Process of work & Operation',
             'GOL': 'Licensed & Professional certificate', 'GOE': 'Economy'}

colors2 = np.array([["#e1f5fe", "#f6c8b6", "#dd191d", "#dd191d", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ff8f00", "#ff8f00", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ffee58", "#ffee58", "#ff8f00", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#ffee58", "#ffee58", "#ff8f00", "#ff8f00"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#42db41", "#42db41", "#ffee58", "#ffee58"],
                    ["#e1f5fe", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6"],
                    ["#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe"]])

risk_color_data = {
    'Category Color': ["Critical", "Critical", "Critical", "Critical", "Critical", "High", "High", "Critical",
                       "Critical", "Critical", "Medium", "Medium", "High", "Critical", "Critical", "Low", "Medium",
                       "Medium", "High", "High", "Low", "Low", "Low", "Medium", "Medium"],
    'Risk Level': ["51", "52", "53", "54", "55", "41", "42", "43", "44", "45", "31", "32", "33", "34", "35", "21", "22",
                   "23", "24", "25", "11", "12", "13", "14", "15"]}
risk_color_df = pd.DataFrame(risk_color_data)
display_cols_common = ['Occurrence Date', 'Incident', 'รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact', 'Impact Level',
                       'รายละเอียดการเกิด', 'Resulting Actions']

# ==============================================================================
# PAGE DISPLAY LOGIC
# ==============================================================================
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False

if not st.session_state.data_ready:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        with st.container():
            LOGO_URL_HEADER = "https://raw.githubusercontent.com/HOIARRTool/hoiarr/refs/heads/main/logo1.png"
            LOGO_HEIGHT_HEADER = 28
            header_text = "HOSPITAL OCCURRENCE/INCIDENT ANALYSIS & RISK REGISTER (HOIA-RR) tool"
            logo_html_tag = f'<img src="{LOGO_URL_HEADER}" style="height: {LOGO_HEIGHT_HEADER}px; margin-right: 10px; vertical-align: middle;">'
            st.markdown(
                f'<div class="custom-header" style="display: flex; align-items: center;">{logo_html_tag}<span>{header_text}</span></div>',
                unsafe_allow_html=True)
        st.title("Welcome")
        st.markdown("Please upload your incident report file (.xlsx) to begin the analysis.")
        if st.session_state.get('upload_error_message'):
            st.error(st.session_state.upload_error_message)
            st.session_state.upload_error_message = None
        uploaded_file_main = st.file_uploader("Choose your Excel file here:", type=".xlsx", key="main_page_uploader",
                                              label_visibility="collapsed")

        st.markdown(
            '<p style="font-size:12px; color:gray;">*This tool is part of the thesis "IMPLEMENTING THE HOSPITAL OCCURRENCE/INCIDENT ANALYSIS & RISK REGISTER (HOIA-RR TOOL) IN THAI HOSPITALS: A STUDY ON EFFECTIVE ADOPTION" by Ms. Vilasinee Kheunkaew, a master\'s degree student at the School of Health Science, Mae Fah Luang University.</p>',
            unsafe_allow_html=True)

    if uploaded_file_main is not None:
        st.session_state.upload_error_message = None
        with st.spinner("Processing file, please wait..."):
            df = load_data(uploaded_file_main)
            if not df.empty:
                df.replace('', 'None', inplace=True)
                df = df.fillna('None')
                # The code expects Thai headers in the input file and renames them. This is intentional.
                df.rename(columns={'วดป.ที่เกิด': 'Occurrence Date', 'ความรุนแรง': 'Impact'}, inplace=True)
                required_cols_in_upload = ['Incident', 'Occurrence Date', 'Impact']
                missing_cols_check = [col for col in required_cols_in_upload if col not in df.columns]
                if missing_cols_check:
                    st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols_check)}.")
                    st.stop()
                df['Impact_original_value'] = df['Impact']
                df['Impact'] = df['Impact'].astype(str).str.strip()
                df['รหัส'] = df['Incident'].astype(str).str.slice(0, 6).str.strip()
                if not df2.empty:
                    df = pd.merge(df, df2, on='รหัส', how='left')
                for col_name in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]:
                    if col_name not in df.columns:
                        df[col_name] = 'N/A (Data from AllCode unavailable)'
                    else:
                        df[col_name].fillna('N/A (Code not found in AllCode)', inplace=True)
                try:
                    df['Occurrence Date'] = pd.to_datetime(df['Occurrence Date'], errors='coerce')
                    df.dropna(subset=['Occurrence Date'], inplace=True)
                    if df.empty:
                        st.error("No valid 'Occurrence Date' data found after conversion.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error converting 'Occurrence Date': {e}")
                    st.stop()

                metrics_data_dict = {}
                metrics_data_dict['df_original_rows'] = df.shape[0]
                metrics_data_dict['total_psg9_incidents_for_metric1'] = \
                    df[df['รหัส'].isin(psg9_r_codes_for_counting)].shape[0] if psg9_r_codes_for_counting else 0
                if sentinel_composite_keys:
                    df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                        str).str.strip()
                    metrics_data_dict['total_sentinel_incidents_for_metric1'] = \
                        df[df['Sentinel code for check'].isin(sentinel_composite_keys)].shape[0]
                else:
                    metrics_data_dict['total_sentinel_incidents_for_metric1'] = 0

                impact_level_map = {('A', 'B', '1'): '1', ('C', 'D', '2'): '2', ('E', 'F', '3'): '3',
                                    ('G', 'H', '4'): '4', ('I', '5'): '5'}


                def map_impact_level_func(impact_val):
                    impact_val_str = str(impact_val)
                    for k, v_level in impact_level_map.items():
                        if impact_val_str in k: return v_level
                    return 'N/A'


                df['Impact Level'] = df['Impact'].apply(map_impact_level_func)
                severe_impact_levels_list = ['3', '4', '5']
                df_severe_incidents_calc = df[df['Impact Level'].isin(severe_impact_levels_list)].copy()
                metrics_data_dict['total_severe_incidents'] = df_severe_incidents_calc.shape[0]
                metrics_data_dict['total_severe_psg9_incidents'] = \
                    df_severe_incidents_calc[df_severe_incidents_calc['รหัส'].isin(psg9_r_codes_for_counting)].shape[
                        0] if psg9_r_codes_for_counting else 0

                if 'Resulting Actions' in df.columns:
                    unresolved_conditions = df_severe_incidents_calc['Resulting Actions'].astype(str).isin(['None', ''])
                    df_severe_unresolved_calc = df_severe_incidents_calc[unresolved_conditions].copy()
                    metrics_data_dict['total_severe_unresolved_incidents_val'] = df_severe_unresolved_calc.shape[0]
                    metrics_data_dict['total_severe_unresolved_psg9_incidents_val'] = \
                        df_severe_unresolved_calc[
                            df_severe_unresolved_calc['รหัส'].isin(psg9_r_codes_for_counting)].shape[
                            0] if psg9_r_codes_for_counting else 0
                else:
                    df_severe_unresolved_calc = pd.DataFrame()
                    metrics_data_dict['total_severe_unresolved_incidents_val'] = "N/A"
                    metrics_data_dict['total_severe_unresolved_psg9_incidents_val'] = "N/A"

                metrics_data_dict['df_severe_unresolved_for_expander'] = df_severe_unresolved_calc
                df.drop(columns=[col for col in
                                 ['In.HCode', 'วดป.ที่ Import การเกิด', 'รหัสรายงาน', 'Sentinel code for check'] if
                                 col in df.columns], inplace=True, errors='ignore')

                total_month_calc = 1
                if not df.empty:
                    max_date_period = df['Occurrence Date'].max().to_period('M')
                    min_date_period = df['Occurrence Date'].min().to_period('M')
                    total_month_calc = (max_date_period.year - min_date_period.year) * 12 + (
                            max_date_period.month - min_date_period.month) + 1
                metrics_data_dict['total_month'] = max(1, total_month_calc)

                df['Incident Type'] = df['Incident'].astype(str).str[:3]
                df['Month'] = df['Occurrence Date'].dt.month
                df['เดือน'] = df['Month'].map(month_label)
                df['Year'] = df['Occurrence Date'].dt.year.astype(str)

                PSG9_ID_COL = 'PSG_ID'
                if not PSG9code_df_master.empty and PSG9_ID_COL in PSG9code_df_master.columns:
                    standards_to_merge = PSG9code_df_master[['รหัส', PSG9_ID_COL]].copy().drop_duplicates(
                        subset=['รหัส'])
                    standards_to_merge['รหัส'] = standards_to_merge['รหัส'].astype(str).str.strip()
                    df = pd.merge(df, standards_to_merge, on='รหัส', how='left')
                    df['หมวดหมู่มาตรฐานสำคัญ'] = df[PSG9_ID_COL].map(PSG9_label_dict).fillna(
                        "Not in PSG9 Catalog")
                else:
                    df['หมวดหมู่มาตรฐานสำคัญ'] = "Could not identify (PSG9code.xlsx not loaded)"

                df_freq_temp = df['Incident'].value_counts().reset_index()
                df_freq_temp.columns = ['Incident', 'count']
                df_freq_temp['Incident Rate/mth'] = (df_freq_temp['count'] / metrics_data_dict['total_month']).round(1)
                df = pd.merge(df, df_freq_temp, on="Incident", how='left')
                st.session_state.df_freq_for_display = df_freq_temp.copy()

                conditions_freq = [(df['Incident Rate/mth'] < 2.0), (df['Incident Rate/mth'] < 3.9),
                                   (df['Incident Rate/mth'] < 6.9), (df['Incident Rate/mth'] < 29.9)]
                choices_freq = ['1', '2', '3', '4']
                df['Frequency Level'] = np.select(conditions_freq, choices_freq, default='5')
                df['Risk Level'] = df.apply(
                    lambda row: f"{row['Impact Level']}{row['Frequency Level']}" if pd.notna(row['Impact Level']) and
                                                                                    row[
                                                                                        'Impact Level'] != 'N/A' else 'N/A',
                    axis=1)
                df = pd.merge(df, risk_color_df, on='Risk Level', how='left')
                df['Category Color'].fillna('Undefined', inplace=True)

                metrics_data_dict['total_processed_incidents'] = df.shape[0]
                st.session_state.processed_df = df.copy()
                st.session_state.metrics_data = metrics_data_dict
                st.session_state.data_ready = True
                st.rerun()
else:
    # ==============================================================================
    # MAIN APPLICATION STRUCTURE
    # ==============================================================================
    df = st.session_state.get('processed_df', pd.DataFrame())
    metrics_data = st.session_state.get('metrics_data', {})
    df_freq = st.session_state.get('df_freq_for_display', pd.DataFrame())

    if df.empty:
        st.error("Data is not available. Please return to the upload page.")
        if st.button("Return to Upload Page"):
            st.session_state.clear()
            st.rerun()
        st.stop()

    total_month = metrics_data.get("total_month", 1)

    st.sidebar.markdown(
        f"""<div style="display: flex; align-items: center; margin-bottom: 1rem;"><img src="{LOGO_URL}" style="height: 32px; margin-right: 10px;"><h2 style="margin: 0; font-size: 1.7rem; color: #001f3f; font-weight: bold;">HOIA-RR Menu</h2></div>""",
        unsafe_allow_html=True)
    if st.sidebar.button("Upload New File", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.sidebar.markdown("---")
    if 'Occurrence Date' in df.columns and not df.empty and pd.api.types.is_datetime64_any_dtype(df['Occurrence Date']):
        min_date_str = df['Occurrence Date'].min().strftime('%m/%Y')
        max_date_str = df['Occurrence Date'].max().strftime('%m/%Y')
        st.sidebar.markdown(f"**Analysis Period:** {min_date_str} to {max_date_str}")
    st.sidebar.markdown(f"**Total Months:** {total_month} months")
    st.sidebar.markdown(f"**Incidents in File:** {df.shape[0]:,} items")
    st.sidebar.markdown("Select an analysis to display:")

    selected_analysis = st.session_state.get('selected_analysis', analysis_options_list[0])
    for option in analysis_options_list:
        is_selected = (selected_analysis == option)
        button_type = "primary" if is_selected else "secondary"
        if st.sidebar.button(option, key=f"btn_{option}", type=button_type, use_container_width=True):
            st.session_state.selected_analysis = option
            st.rerun()

    st.sidebar.markdown("---")
    line_oa_id = "@144pdywm"
    line_url = f"https://page.line.me/{line_oa_id}"
    st.sidebar.link_button("Chat with IR-Chatbot", line_url, use_container_width=True, type="secondary")
    st.sidebar.markdown("(For severity level analysis and contributing factor recommendations per incident)")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        **Acknowledgements:** Special thanks to 
        - Dr. Anuwat Supachutikul
        - Dr. Kongkiat Kespetch
        - Dr. Piyawan Limpanyalert
        - Prm. Poramin Weerananthawat
        - Asst. Prof. Dr. Niwes Kulwong (Advisor)

        for their initiative, contributions, support, and inspiration, which were foundational to the development of this tool.
        """)

    # =========================================================================
    # PAGE CONTENT ROUTING
    # =========================================================================
    if selected_analysis == "Overall Summary Dashboard":
        st.markdown("<h4 style='color: #001f3f;'>Incident Summary Dashboard:</h4>", unsafe_allow_html=True)

        with st.expander("Show/Hide Full Data Table"):
            st.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "Occurrence Date": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY")
            })

        dashboard_expander_cols = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด', 'Resulting Actions']
        date_format_config = {
            "Occurrence Date": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY")
        }

        total_processed_incidents = metrics_data.get("total_processed_incidents", 0)
        total_psg9_incidents_for_metric1 = metrics_data.get("total_psg9_incidents_for_metric1", 0)
        total_sentinel_incidents_for_metric1 = metrics_data.get("total_sentinel_incidents_for_metric1", 0)
        total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
        total_severe_psg9_incidents = metrics_data.get("total_severe_psg9_incidents", 0)
        total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", "N/A")
        total_severe_unresolved_psg9_incidents_val = metrics_data.get("total_severe_unresolved_psg9_incidents_val",
                                                                      "N/A")
        df_severe_incidents = df[
            df['Impact Level'].isin(['3', '4', '5'])].copy() if 'Impact Level' in df.columns else pd.DataFrame()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{total_processed_incidents:,}")
        with col2:
            st.metric("PSG9", f"{total_psg9_incidents_for_metric1:,}")
            with st.expander(f"View Details ({total_psg9_incidents_for_metric1} items)"):
                psg9_df = df[
                    df['รหัส'].isin(psg9_r_codes_for_counting)] if psg9_r_codes_for_counting else pd.DataFrame()
                st.dataframe(psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col3:
            st.metric("Sentinel", f"{total_sentinel_incidents_for_metric1:,}")
            with st.expander(f"View Details ({total_sentinel_incidents_for_metric1} items)"):
                if sentinel_composite_keys:
                    df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                        str).str.strip()
                    sentinel_df = df[df['Sentinel code for check'].isin(sentinel_composite_keys)]
                    st.dataframe(sentinel_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)
                else:
                    st.write("Could not display data.")

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            st.metric("E-I & 3-5 [all]", f"{total_severe_incidents:,}")
            with st.expander(f"View Details ({total_severe_incidents} items)"):
                st.dataframe(df_severe_incidents[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col5:
            st.metric("E-I & 3-5 [PSG9]", f"{total_severe_psg9_incidents:,}")
            with st.expander(f"View Details ({total_severe_psg9_incidents} items)"):
                severe_psg9_df = df_severe_incidents[df_severe_incidents['รหัส'].isin(
                    psg9_r_codes_for_counting)] if psg9_r_codes_for_counting else pd.DataFrame()
                st.dataframe(severe_psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col6:
            val_unresolved_all = f"{total_severe_unresolved_incidents_val:,}" if isinstance(
                total_severe_unresolved_incidents_val, int) else "N/A"
            st.metric(f"E-I & 3-5 [all] Unresolved", val_unresolved_all)
            if isinstance(total_severe_unresolved_incidents_val, int) and total_severe_unresolved_incidents_val > 0:
                with st.expander(f"View Details ({total_severe_unresolved_incidents_val} items)"):
                    unresolved_df_all = df[
                        df['Impact Level'].isin(['3', '4', '5']) & df['Resulting Actions'].astype(str).isin(
                            ['None', ''])]
                    st.dataframe(unresolved_df_all[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)
        with col7:
            val_unresolved_psg9 = f"{total_severe_unresolved_psg9_incidents_val:,}" if isinstance(
                total_severe_unresolved_psg9_incidents_val, int) else "N/A"
            st.metric(f"E-I & 3-5 [PSG9] Unresolved", val_unresolved_psg9)
            if isinstance(total_severe_unresolved_psg9_incidents_val,
                          int) and total_severe_unresolved_psg9_incidents_val > 0:
                with st.expander(f"View Details ({total_severe_unresolved_psg9_incidents_val} items)"):
                    unresolved_df_all = df[
                        df['Impact Level'].isin(['3', '4', '5']) & df['Resulting Actions'].astype(str).isin(
                            ['None', ''])]
                    unresolved_df_psg9 = unresolved_df_all[unresolved_df_all['รหัส'].isin(
                        psg9_r_codes_for_counting)] if psg9_r_codes_for_counting else pd.DataFrame()
                    st.dataframe(unresolved_df_psg9[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)

    elif selected_analysis == "Sentinel Events List":
        st.markdown("<h4 style='color: #001f3f;'>Detected Sentinel Events</h4>", unsafe_allow_html=True)
        if sentinel_composite_keys:
            df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                str).str.strip()
            sentinel_events = df[df['Sentinel code for check'].isin(sentinel_composite_keys)].copy()
            if not Sentinel2024_df.empty and 'ชื่ออุบัติการณ์ความเสี่ยง' in Sentinel2024_df.columns:
                sentinel_events = pd.merge(sentinel_events,
                                           Sentinel2024_df[['รหัส', 'Impact', 'ชื่ออุบัติการณ์ความเสี่ยง']].rename(
                                               columns={'ชื่ออุบัติการณ์ความเสี่ยง': 'Sentinel Event Name'}),
                                           on=['รหัส', 'Impact'], how='left')
            display_sentinel_cols = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด', 'Resulting Actions']
            if 'Sentinel Event Name' in sentinel_events.columns:
                display_sentinel_cols.insert(2, 'Sentinel Event Name')
            final_display_cols = [col for col in display_sentinel_cols if col in sentinel_events.columns]
            st.dataframe(sentinel_events[final_display_cols], use_container_width=True, hide_index=True,
                         column_config={
                             "Occurrence Date": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY")})
        else:
            st.warning("Could not check for Sentinel Events (Sentinel2024.xlsx may be empty or missing).")

    elif selected_analysis == "Interactive Risk Matrix":
        st.subheader("Interactive Risk Matrix")
        matrix_data_counts = np.zeros((5, 5), dtype=int)
        impact_level_keys = ['5', '4', '3', '2', '1']
        freq_level_keys = ['1', '2', '3', '4', '5']
        if 'Risk Level' in df.columns and 'Impact Level' in df.columns and 'Frequency Level' in df.columns and not df[
            df['Risk Level'] != 'N/A'].empty:
            risk_counts_df = df.groupby(['Impact Level', 'Frequency Level']).size().reset_index(name='counts')
            for _, row in risk_counts_df.iterrows():
                il_key = str(row['Impact Level'])
                fl_key = str(row['Frequency Level'])
                count_val = row['counts']
                if il_key in impact_level_keys and fl_key in freq_level_keys:
                    row_idx = impact_level_keys.index(il_key)
                    col_idx = freq_level_keys.index(fl_key)
                    matrix_data_counts[row_idx, col_idx] = count_val

        impact_labels_display = {
            '5': "I / 5<br>Extreme / Death", '4': "G-H / 4<br>Major / Severe",
            '3': "E-F / 3<br>Moderate", '2': "C-D / 2<br>Minor / Low", '1': "A-B / 1<br>Insignificant / No Harm"
        }
        freq_labels_display_short = {"1": "F1", "2": "F2", "3": "F3", "4": "F4", "5": "F5"}
        freq_labels_display_long = {
            "1": "Remote<br>(<2/mth)", "2": "Uncommon<br>(2-3/mth)", "3": "Occasional<br>(4-6/mth)",
            "4": "Probable<br>(7-29/mth)", "5": "Frequent<br>(>=30/mth)"
        }
        impact_to_color_row = {'5': 0, '4': 1, '3': 2, '2': 3, '1': 4}
        freq_to_color_col = {'1': 2, '2': 3, '3': 4, '4': 5, '5': 6}
        cols_header = st.columns([2.2, 1, 1, 1, 1, 1])
        with cols_header[0]:
            st.markdown(
                f"<div style='background-color:{colors2[6, 0]}; color:{get_text_color_for_bg(colors2[6, 0])}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; align-items:center; justify-content:center;'>Impact / Frequency</div>",
                unsafe_allow_html=True)
        for i, fl_key in enumerate(freq_level_keys):
            with cols_header[i + 1]:
                header_freq_bg_color = colors2[5, freq_to_color_col.get(fl_key, 2) - 1]
                header_freq_text_color = get_text_color_for_bg(header_freq_bg_color)
                st.markdown(
                    f"<div style='background-color:{header_freq_bg_color}; color:{header_freq_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; flex-direction: column; align-items:center; justify-content:center;'><div>{freq_labels_display_short.get(fl_key, '')}</div><div style='font-size:0.7em;'>{freq_labels_display_long.get(fl_key, '')}</div></div>",
                    unsafe_allow_html=True)

        for il_key in impact_level_keys:
            cols_data_row = st.columns([2.2, 1, 1, 1, 1, 1])
            row_idx_color = impact_to_color_row[il_key]
            with cols_data_row[0]:
                header_impact_bg_color = colors2[row_idx_color, 1]
                header_impact_text_color = get_text_color_for_bg(header_impact_bg_color)
                st.markdown(
                    f"<div style='background-color:{header_impact_bg_color}; color:{header_impact_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:70px; display:flex; align-items:center; justify-content:center;'>{impact_labels_display[il_key]}</div>",
                    unsafe_allow_html=True)
            for i, fl_key in enumerate(freq_level_keys):
                with cols_data_row[i + 1]:
                    count = matrix_data_counts[impact_level_keys.index(il_key), freq_level_keys.index(fl_key)]
                    cell_bg_color = colors2[row_idx_color, freq_to_color_col[fl_key]]
                    text_color = get_text_color_for_bg(cell_bg_color)
                    st.markdown(
                        f"<div style='background-color:{cell_bg_color}; color:{text_color}; padding:5px; margin:1px; border-radius:3px; text-align:center; font-weight:bold; min-height:40px; display:flex; align-items:center; justify-content:center;'>{count}</div>",
                        unsafe_allow_html=True)
                    if count > 0:
                        button_key = f"view_risk_{il_key}_{fl_key}"
                        if st.button("👁️", key=button_key, help=f"View - {count} items", use_container_width=True):
                            st.session_state.clicked_risk_impact = il_key
                            st.session_state.clicked_risk_freq = fl_key
                            st.session_state.show_incident_table = True
                            st.rerun()
                    else:
                        st.markdown("<div style='height:38px; margin-top:5px;'></div>", unsafe_allow_html=True)

        if st.session_state.show_incident_table and st.session_state.clicked_risk_impact is not None:
            il_selected = st.session_state.clicked_risk_impact
            fl_selected = st.session_state.clicked_risk_freq
            df_filtered_incidents = df[(df['Impact Level'].astype(str) == il_selected) & (
                    df['Frequency Level'].astype(str) == fl_selected)].copy()
            expander_title = f"Incident List: Impact Level {il_selected}, Frequency Level {fl_selected} - Found {len(df_filtered_incidents)} items"
            with st.expander(expander_title, expanded=True):
                st.dataframe(df_filtered_incidents[display_cols_common], use_container_width=True, hide_index=True)
                if st.button("Close List", key="clear_risk_selection_button"):
                    st.session_state.show_incident_table = False
                    st.session_state.clicked_risk_impact = None
                    st.session_state.clicked_risk_freq = None
                    st.rerun()

    elif selected_analysis == "Incident Resolution":
        st.subheader("✅ Resolved Incidents Summary (by Category)")
        all_categories = sorted([cat for cat in df['หมวด'].unique() if cat and pd.notna(cat)])
        resolved_df_total_count = df[~df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])].copy()
        if resolved_df_total_count.empty:
            st.info("No resolved incidents have been recorded yet.")
        else:
            st.metric("Total Resolved Incidents", f"{len(resolved_df_total_count):,} items")
            for category in all_categories:
                total_in_category_df = df[df['หมวด'] == category]
                total_count = len(total_in_category_df)
                resolved_in_category_df = total_in_category_df[
                    ~total_in_category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                resolved_count = len(resolved_in_category_df)
                if resolved_count > 0:
                    expander_title = f"Category: {category} (Resolved {resolved_count}/{total_count} items)"
                    with st.expander(expander_title):
                        display_cols = ['Occurrence Date', 'Incident', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact',
                                        'Resulting Actions']
                        displayable_cols = [col for col in display_cols if col in resolved_in_category_df.columns]
                        st.dataframe(resolved_in_category_df[displayable_cols], use_container_width=True,
                                     hide_index=True)
        st.markdown("---")
        st.subheader("⏱️ Unresolved Incidents (by Severity)")
        unresolved_df = df[df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])].copy()
        if unresolved_df.empty:
            st.success("🎉 No unresolved items found. Excellent!")
        else:
            st.metric("Total Unresolved Incidents", f"{len(unresolved_df):,} items")
            severity_order = ['Critical', 'High', 'Medium', 'Low', 'Undefined']
            for severity in severity_order:
                severity_df = unresolved_df[unresolved_df['Category Color'] == severity]
                if not severity_df.empty:
                    with st.expander(f"Severity Level: {severity} ({len(severity_df)} items)"):
                        display_cols = ['Occurrence Date', 'Incident', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact',
                                        'Impact Level', 'รายละเอียดการเกิด']
                        displayable_cols = [col for col in display_cols if col in severity_df.columns]
                        st.dataframe(severity_df[displayable_cols], use_container_width=True, hide_index=True)

    elif selected_analysis == "Incident Summary Graphs":
        st.markdown("<h4 style='color: #001f3f;'>Incident Summary Graphs (by Dimension)</h4>", unsafe_allow_html=True)
        pastel_color_discrete_map_dimensions = {
            'Critical': '#FF9999', 'High': '#FFCC99', 'Medium': '#FFFF99',
            'Low': '#99FF99', 'Undefined': '#D3D3D3'
        }
        tab1_v, tab2_v, tab3_v, tab4_v = st.tabs(
            ["👁️ By Goals (Category)", "👁️ By Group", "👁️ By Shift", "👁️ By Place"])
        if isinstance(df, pd.DataFrame):
            df_charts = df.copy()
            if 'Count' not in df_charts.columns: df_charts['Count'] = 1

            with tab1_v:
                st.markdown(f"Incidents by Safety Goals ({total_month}m)")
                if 'หมวด' in df_charts.columns:
                    df_c1 = df_charts[~df_charts['หมวด'].isin(
                        ['N/A (Data from AllCode unavailable)', 'N/A (Code not found in AllCode)'])]
                    if not df_c1.empty:
                        fig_c1 = px.bar(df_c1, x='หมวด', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c1, use_container_width=True)
            with tab2_v:
                st.markdown(f"Incidents by Group ({total_month}m)")
                if 'กลุ่ม' in df_charts.columns:
                    df_c2 = df_charts[~df_charts['กลุ่ม'].isin(
                        ['N/A (Data from AllCode unavailable)', 'N/A (Code not found in AllCode)'])]
                    if not df_c2.empty:
                        fig_c2 = px.bar(df_c2, x='กลุ่ม', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c2, use_container_width=True)
            with tab3_v:
                st.markdown(f"Incidents by Shift ({total_month}m)")
                if 'ช่วงเวลา/เวร' in df_charts.columns:
                    df_c3 = df_charts[df_charts['ช่วงเวลา/เวร'] != 'N/A (Data from uploaded file)']
                    if not df_c3.empty:
                        fig_c3 = px.bar(df_c3, x='ช่วงเวลา/เวร', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c3, use_container_width=True)
            with tab4_v:
                st.markdown(f"Incidents by Place ({total_month}m)")
                if 'ชนิดสถานที่' in df_charts.columns:
                    df_c4 = df_charts[df_charts['ชนิดสถานที่'] != 'N/A (Data from uploaded file)']
                    if not df_c4.empty:
                        fig_c4 = px.bar(df_c4, x='ชนิดสถานที่', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c4, use_container_width=True)

    elif selected_analysis == "Sankey: Overall View":
        st.markdown("<h4 style='color: #001f3f;'>Sankey Diagram: Overall View</h4>", unsafe_allow_html=True)
        st.markdown("""
        <style>
            .plot-container .svg-container .sankey-node text {
                stroke-width: 0 !important; text-shadow: none !important; paint-order: stroke fill;
            }
        </style>
        """, unsafe_allow_html=True)
        req_cols = ['หมวด', 'Impact', 'Impact Level', 'Category Color']
        if isinstance(df, pd.DataFrame) and all(col in df.columns for col in req_cols):
            sankey_df = df.copy()
            placeholders_to_filter_generic = ['None', '', np.nan, 'N/A', 'ไม่ระบุ', 'Not Specified',
                                              'N/A (Data from AllCode unavailable)', 'N/A (Code not found in AllCode)']
            placeholders_to_filter_stripped_lower = [str(p).strip().lower() for p in placeholders_to_filter_generic if
                                                     pd.notna(p)]

            sankey_df['หมวด_Node'] = "Category: " + sankey_df['หมวด'].astype(str).str.strip()
            sankey_df = sankey_df[~sankey_df['หมวด'].str.lower().isin(placeholders_to_filter_stripped_lower)]
            sankey_df = sankey_df[sankey_df['หมวด'] != '']

            sankey_df['Impact_AI_Node'] = "Impact: " + sankey_df['Impact'].astype(str).str.strip()
            sankey_df.loc[sankey_df['Impact'].str.lower().isin(placeholders_to_filter_stripped_lower) | (
                        sankey_df['Impact'] == ''), 'Impact_AI_Node'] = "Impact: Not Specified A-I"

            impact_level_display_map = {'1': "Level: 1 (A-B)", '2': "Level: 2 (C-D)", '3': "Level: 3 (E-F)",
                                        '4': "Level: 4 (G-H)", '5': "Level: 5 (I)", 'N/A': "Level: Not Specified"}
            sankey_df['Impact_Level_Node'] = sankey_df['Impact Level'].astype(str).str.strip().map(
                impact_level_display_map).fillna("Level: Not Specified")
            sankey_df['Risk_Category_Node'] = "Risk: " + sankey_df['Category Color'].astype(str).str.strip()

            node_cols = ['หมวด_Node', 'Impact_AI_Node', 'Impact_Level_Node', 'Risk_Category_Node']
            sankey_df.dropna(subset=node_cols, inplace=True)

            if sankey_df.empty:
                st.warning("No data available to display in the Sankey diagram after filtering.")
            else:
                labels_muad = sorted(list(sankey_df['หมวด_Node'].unique()))
                impact_ai_order = [f"Impact: {i}" for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']] + [
                    "Impact: Not Specified A-I"]
                labels_impact_ai = sorted(list(sankey_df['Impact_AI_Node'].unique()),
                                          key=lambda x: impact_ai_order.index(x) if x in impact_ai_order else 999)
                level_order_map = {"Level: 1 (A-B)": 1, "Level: 2 (C-D)": 2, "Level: 3 (E-F)": 3, "Level: 4 (G-H)": 4,
                                   "Level: 5 (I)": 5, "Level: Not Specified": 6}
                labels_impact_level = sorted(list(sankey_df['Impact_Level_Node'].unique()),
                                             key=lambda x: level_order_map.get(x, 999))
                risk_order = ["Risk: Critical", "Risk: High", "Risk: Medium", "Risk: Low", "Risk: Undefined"]
                labels_risk_cat = sorted(list(sankey_df['Risk_Category_Node'].unique()),
                                         key=lambda x: risk_order.index(x) if x in risk_order else 999)

                all_labels_ordered = labels_muad + labels_impact_ai + labels_impact_level + labels_risk_cat
                all_labels = list(pd.Series(all_labels_ordered).unique())
                label_to_idx = {label: i for i, label in enumerate(all_labels)}

                source_indices, target_indices, values = [], [], []
                links1 = sankey_df.groupby(['หมวด_Node', 'Impact_AI_Node']).size().reset_index(name='value')
                for _, row in links1.iterrows():
                    source_indices.append(label_to_idx[row['หมวด_Node']])
                    target_indices.append(label_to_idx[row['Impact_AI_Node']])
                    values.append(row['value'])
                links2 = sankey_df.groupby(['Impact_AI_Node', 'Impact_Level_Node']).size().reset_index(name='value')
                for _, row in links2.iterrows():
                    source_indices.append(label_to_idx[row['Impact_AI_Node']])
                    target_indices.append(label_to_idx[row['Impact_Level_Node']])
                    values.append(row['value'])
                links3 = sankey_df.groupby(['Impact_Level_Node', 'Risk_Category_Node']).size().reset_index(name='value')
                for _, row in links3.iterrows():
                    source_indices.append(label_to_idx[row['Impact_Level_Node']])
                    target_indices.append(label_to_idx[row['Risk_Category_Node']])
                    values.append(row['value'])

                if source_indices:
                    node_colors = []
                    palette1, palette2, palette3 = px.colors.qualitative.Pastel1, px.colors.qualitative.Pastel2, px.colors.qualitative.Set3
                    risk_color_map = {"Risk: Critical": "red", "Risk: High": "orange", "Risk: Medium": "#F7DC6F",
                                      "Risk: Low": "green", "Risk: Undefined": "grey"}
                    for label in all_labels:
                        if label in labels_muad:
                            node_colors.append(palette1[labels_muad.index(label) % len(palette1)])
                        elif label in labels_impact_ai:
                            node_colors.append(palette2[labels_impact_ai.index(label) % len(palette2)])
                        elif label in labels_impact_level:
                            node_colors.append(palette3[labels_impact_level.index(label) % len(palette3)])
                        elif label in labels_risk_cat:
                            node_colors.append(risk_color_map.get(label, 'grey'))
                        else:
                            node_colors.append('rgba(200,200,200,0.8)')
                    link_colors_rgba = [
                        f'rgba({int(c.lstrip("#")[0:2], 16)},{int(c.lstrip("#")[2:4], 16)},{int(c.lstrip("#")[4:6], 16)},0.3)' if c.startswith(
                            '#') else 'rgba(200,200,200,0.3)' for c in [node_colors[s] for s in source_indices]]

                    fig = go.Figure(data=[go.Sankey(
                        arrangement='snap',
                        node=dict(pad=15, thickness=18, line=dict(color="rgba(0,0,0,0.6)", width=0.75),
                                  label=all_labels, color=node_colors,
                                  hovertemplate='%{label} Count: %{value}<extra></extra>'),
                        link=dict(source=source_indices, target=target_indices, value=values, color=link_colors_rgba,
                                  hovertemplate='From %{source.label}<br />To %{target.label}<br />Count: %{value}<extra></extra>')
                    )])
                    fig.update_layout(
                        title_text="<b>Sankey Diagram:</b> Category -> Impact (A-I) -> Impact Level (1-5) -> Risk Category",
                        font_size=12, height=max(700, len(all_labels) * 18), width=1200, template='plotly_white',
                        margin=dict(t=60, l=10, r=10, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not create links for the Sankey diagram.")
        else:
            st.warning(f"Missing required columns ({', '.join(req_cols)}) for creating the Sankey diagram.")

    elif selected_analysis == "Sankey: 9 Patient Safety Goals":
        st.markdown("<h4 style='color: #001f3f;'>Sankey Diagram: 9 Patient Safety Goals (PSG9)</h4>",
                    unsafe_allow_html=True)
        st.markdown("""
        <style>
            .plot-container .svg-container .sankey-node text {
                stroke-width: 0 !important; text-shadow: none !important; paint-order: stroke fill;
            }
        </style>
        """, unsafe_allow_html=True)

        psg9_placeholders_to_filter = ["Not in PSG9 Catalog", "Could not identify (Merge PSG9 failed)"]
        psg9_placeholders_stripped_lower = [str(p).strip().lower() for p in psg9_placeholders_to_filter if pd.notna(p)]
        required_cols_sankey_simplified = ['หมวดหมู่มาตรฐานสำคัญ', 'รหัส', 'Impact', 'ชื่ออุบัติการณ์ความเสี่ยง',
                                           'Category Color']

        if isinstance(df, pd.DataFrame) and all(col in df.columns for col in required_cols_sankey_simplified):
            sankey_df_new = df.copy()
            sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned'] = sankey_df_new['หมวดหมู่มาตรฐานสำคัญ'].astype(
                str).str.strip()
            sankey_df_new = sankey_df_new[
                ~sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned'].str.lower().isin(psg9_placeholders_stripped_lower)]

            psg9_to_cp_gp_map = {PSG9_label_dict[num].strip(): 'CP (PSG Category)' for num in [1, 3, 4, 5, 6, 7, 8, 9]
                                 if num in PSG9_label_dict}
            psg9_to_cp_gp_map.update(
                {PSG9_label_dict[num].strip(): 'GP (PSG Category)' for num in [2] if num in PSG9_label_dict})

            sankey_df_new['หมวด_CP_GP_Node'] = sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned'].map(psg9_to_cp_gp_map)
            sankey_df_new['หมวดหมู่PSG_Node'] = "PSG9: " + sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned']
            sankey_df_new['รหัส_Node'] = "Code: " + sankey_df_new['รหัส'].astype(str).str.strip()
            sankey_df_new['Impact_AI_Node'] = "Impact: " + sankey_df_new['Impact'].astype(str).str.strip()
            sankey_df_new['Risk_Category_Node'] = "Risk: " + sankey_df_new['Category Color'].fillna('Undefined')
            sankey_df_new['ชื่ออุบัติการณ์ความเสี่ยง_for_hover'] = sankey_df_new['ชื่ออุบัติการณ์ความเสี่ยง'].fillna(
                'No description')

            cols_for_dropna_new_sankey = ['หมวด_CP_GP_Node', 'หมวดหมู่PSG_Node', 'รหัส_Node', 'Impact_AI_Node',
                                          'Risk_Category_Node']
            sankey_df_new.dropna(subset=cols_for_dropna_new_sankey, inplace=True)

            if sankey_df_new.empty:
                st.warning("No PSG9 data available to display in the Sankey diagram after filtering.")
            else:
                labels_muad_cp_gp_simp = sorted(list(sankey_df_new['หมวด_CP_GP_Node'].unique()))
                labels_psg9_cat_simp = sorted(list(sankey_df_new['หมวดหมู่PSG_Node'].unique()))
                rh_node_to_desc_map = sankey_df_new.drop_duplicates(subset=['รหัส_Node']).set_index('รหัส_Node')[
                    'ชื่ออุบัติการณ์ความเสี่ยง_for_hover'].to_dict()
                labels_rh_simp = sorted(list(sankey_df_new['รหัส_Node'].unique()))
                labels_impact_ai_simp = sorted(list(sankey_df_new['Impact_AI_Node'].unique()))
                risk_order = ["Risk: Critical", "Risk: High", "Risk: Medium", "Risk: Low", "Risk: Undefined"]
                labels_risk_category = sorted(list(sankey_df_new['Risk_Category_Node'].unique()),
                                              key=lambda x: risk_order.index(x) if x in risk_order else 99)
                all_labels_ordered_simp = labels_muad_cp_gp_simp + labels_psg9_cat_simp + labels_rh_simp + labels_impact_ai_simp + labels_risk_category
                all_labels_simp = list(pd.Series(all_labels_ordered_simp).unique())
                label_to_idx_simp = {label: i for i, label in enumerate(all_labels_simp)}
                customdata_for_nodes_simp = [
                    f"<br>Description: {str(rh_node_to_desc_map.get(label_node, ''))}" if label_node in rh_node_to_desc_map else ""
                    for label_node in all_labels_simp]

                source_indices_simp, target_indices_simp, values_simp = [], [], []
                links_l1 = sankey_df_new.groupby(['หมวด_CP_GP_Node', 'หมวดหมู่PSG_Node']).size().reset_index(
                    name='value')
                for _, row in links_l1.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['หมวด_CP_GP_Node']]);
                    target_indices_simp.append(label_to_idx_simp[row['หมวดหมู่PSG_Node']]);
                    values_simp.append(row['value'])
                links_l2 = sankey_df_new.groupby(['หมวดหมู่PSG_Node', 'รหัส_Node']).size().reset_index(name='value')
                for _, row in links_l2.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['หมวดหมู่PSG_Node']]);
                    target_indices_simp.append(label_to_idx_simp[row['รหัส_Node']]);
                    values_simp.append(row['value'])
                links_l3 = sankey_df_new.groupby(['รหัส_Node', 'Impact_AI_Node']).size().reset_index(name='value')
                for _, row in links_l3.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['รหัส_Node']]);
                    target_indices_simp.append(label_to_idx_simp[row['Impact_AI_Node']]);
                    values_simp.append(row['value'])
                links_l4 = sankey_df_new.groupby(['Impact_AI_Node', 'Risk_Category_Node']).size().reset_index(
                    name='value')
                for _, row in links_l4.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['Impact_AI_Node']]);
                    target_indices_simp.append(label_to_idx_simp[row['Risk_Category_Node']]);
                    values_simp.append(row['value'])

                if source_indices_simp:
                    node_colors_simp = []
                    palette_l1, palette_l2, palette_l3, palette_l4 = px.colors.qualitative.Bold, px.colors.qualitative.Pastel, px.colors.qualitative.Vivid, px.colors.qualitative.Set3
                    risk_cat_color_map = {"Risk: Critical": "red", "Risk: High": "orange", "Risk: Medium": "#F7DC6F",
                                          "Risk: Low": "green", "Risk: Undefined": "grey"}
                    for label_node in all_labels_simp:
                        if label_node in labels_muad_cp_gp_simp:
                            node_colors_simp.append(
                                palette_l1[labels_muad_cp_gp_simp.index(label_node) % len(palette_l1)])
                        elif label_node in labels_psg9_cat_simp:
                            node_colors_simp.append(
                                palette_l2[labels_psg9_cat_simp.index(label_node) % len(palette_l2)])
                        elif label_node in labels_rh_simp:
                            node_colors_simp.append(palette_l3[labels_rh_simp.index(label_node) % len(palette_l3)])
                        elif label_node in labels_impact_ai_simp:
                            node_colors_simp.append(
                                palette_l4[labels_impact_ai_simp.index(label_node) % len(palette_l4)])
                        elif label_node in labels_risk_category:
                            node_colors_simp.append(risk_cat_color_map.get(label_node, 'grey'))
                        else:
                            node_colors_simp.append('rgba(200,200,200,0.8)')
                    link_colors_simp = []
                    default_link_color_simp = 'rgba(200,200,200,0.35)'
                    for s_idx in source_indices_simp:
                        try:
                            hex_color = node_colors_simp[s_idx];
                            h = hex_color.lstrip('#');
                            rgb_tuple = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
                            link_colors_simp.append(f'rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},0.3)')
                        except:
                            link_colors_simp.append(default_link_color_simp)

                    fig_sankey_psg9_simplified = go.Figure(data=[go.Sankey(
                        arrangement='snap',
                        node=dict(pad=10, thickness=15, line=dict(color="rgba(0,0,0,0.4)", width=0.4),
                                  label=all_labels_simp, color=node_colors_simp, customdata=customdata_for_nodes_simp,
                                  hovertemplate='<b>%{label}</b><br>Count: %{value}%{customdata}<extra></extra>'),
                        link=dict(source=source_indices_simp, target=target_indices_simp, value=values_simp,
                                  color=link_colors_simp,
                                  hovertemplate='From %{source.label}<br />To %{target.label}<br />Count: %{value}<extra></extra>')
                    )])
                    fig_sankey_psg9_simplified.update_layout(
                        title_text="<b>Sankey Diagram:</b> CP/GP -> PSG9 Category -> Code -> Impact -> Risk Category",
                        font_size=11, height=max(800, len(all_labels_simp) * 12 + 200), width=1200,
                        template='plotly_white', margin=dict(t=70, l=10, r=10, b=20))
                    st.plotly_chart(fig_sankey_psg9_simplified, use_container_width=True)
                else:
                    st.warning("Could not create links for the PSG9 Sankey diagram.")
        else:
            st.warning(
                f"Missing required columns ({', '.join(required_cols_sankey_simplified)}) for creating the Sankey diagram.")

    elif selected_analysis == "Scatter Plot & Top 10":
        st.markdown("<h4 style='color: #001f3f;'>Relationship between Incident Code and Category (Scatter Plot)</h4>",
                    unsafe_allow_html=True)
        scatter_cols = ['รหัส', 'หมวด', 'Category Color', 'Incident Rate/mth']
        if isinstance(df, pd.DataFrame) and all(col in df.columns for col in scatter_cols):
            df_sc = df.dropna(subset=scatter_cols, how='any')
            if not df_sc.empty:
                fig_sc = px.scatter(df_sc, x='รหัส', y='หมวด', color='Category Color', size='Incident Rate/mth',
                                    hover_data=['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง'], size_max=30,
                                    color_discrete_map=color_discrete_map)
                st.plotly_chart(fig_sc, theme="streamlit", use_container_width=True)
            else:
                st.warning("Insufficient data for Scatter Plot.")
        else:
            st.warning(f"Missing required columns for Scatter Plot: {scatter_cols}")
        st.markdown("---")
        st.subheader("Top 10 Incidents (by Frequency)")
        if not df_freq.empty:
            df_freq_top10 = df_freq.nlargest(10, 'count')
            st.dataframe(df_freq_top10, use_container_width=True)
        else:
            st.warning("Cannot display Top 10 Incidents.")

    elif selected_analysis == "Summary by Safety Goals":
        st.markdown("<h4 style='color: #001f3f;'>Summary by Safety Goals</h4>", unsafe_allow_html=True)
        goal_definitions = {
            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
            "Personnel Safety": "P:Personnel Safety Goals",
            "Organization Safety": "O:Organization Safety Goals"
        }
        for display_name, cat_name in goal_definitions.items():
            st.markdown(f"##### {display_name}")
            is_org_safety = (display_name == "Organization Safety")
            summary_table = create_goal_summary_table(
                df, cat_name,
                e_up_non_numeric_levels_param=[] if is_org_safety else ['A', 'B', 'C', 'D'],
                e_up_numeric_levels_param=['1', '2'] if is_org_safety else None,
                is_org_safety_table=is_org_safety
            )
            if summary_table is not None and not summary_table.empty:
                st.dataframe(summary_table, use_container_width=True)
            else:
                st.info(f"No data available for '{display_name}'")

    elif selected_analysis == "Analysis by Category & Status":
        st.markdown("<h4 style='color: #001f3f;'>Analysis by Category and Resolution Status</h4>",
                    unsafe_allow_html=True)
        if 'Resulting Actions' not in df.columns or 'หมวดหมู่มาตรฐานสำคัญ' not in df.columns:
            st.error("Cannot display data. Missing 'Resulting Actions' or 'หมวดหมู่มาตรฐานสำคัญ' columns.")
        else:
            tab_psg9, tab_groups, tab_summary = st.tabs(
                ["👁️ Analysis by PSG9 Category", "👁️ Analysis by Main Group (C/G)",
                 "👁️ Severe Incident Resolution Summary (E-I & 3-5)"])

            with tab_psg9:
                st.subheader("Overview of Incidents by Patient Safety Goals (PSG9)")
                psg9_summary_table = create_psg9_summary_table(df)
                if psg9_summary_table is not None and not psg9_summary_table.empty:
                    st.dataframe(psg9_summary_table, use_container_width=True)
                else:
                    st.info("No incident data related to the 9 Patient Safety Goals found.")
                st.markdown("---")
                st.subheader("Resolution Status in each PSG9 Category")
                psg9_categories = {k: v for k, v in PSG9_label_dict.items() if v in df['หมวดหมู่มาตรฐานสำคัญ'].unique()}
                for psg9_id, psg9_name in psg9_categories.items():
                    psg9_df = df[df['หมวดหมู่มาตรฐานสำคัญ'] == psg9_name]
                    total_count = len(psg9_df)
                    resolved_df = psg9_df[~psg9_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                    resolved_count = len(resolved_df)
                    unresolved_count = total_count - resolved_count
                    expander_title = f"{psg9_name} (Total: {total_count} | Resolved: {resolved_count} | Unresolved: {unresolved_count})"
                    with st.expander(expander_title):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Count", f"{total_count:,}")
                        c2.metric("Resolved", f"{resolved_count:,}")
                        c3.metric("Unresolved", f"{unresolved_count:,}")
                        if total_count > 0:
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"Resolved List ({resolved_count})", f"Unresolved List ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("Date",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.info("No resolved items in this category.")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(
                                        psg9_df[psg9_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])][
                                            ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("Date",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.success("All incidents in this category have been resolved.")

            with tab_groups:
                st.subheader("Deep Dive into Resolution Status by Main Group and Subcategory")
                st.markdown("#### Clinical Incidents Group (Code starting with C)")
                df_clinical = df[df['รหัส'].str.startswith('C', na=False)].copy()
                if df_clinical.empty:
                    st.info("No Clinical incident data found.")
                else:
                    clinical_categories = sorted([cat for cat in df_clinical['หมวด'].unique() if cat and pd.notna(cat)])
                    for category in clinical_categories:
                        category_df = df_clinical[df_clinical['หมวด'] == category]
                        total_count = len(category_df);
                        resolved_df = category_df[
                            ~category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])];
                        resolved_count = len(resolved_df);
                        unresolved_count = total_count - resolved_count
                        expander_title = f"{category} (Total: {total_count} | Resolved: {resolved_count} | Unresolved: {unresolved_count})"
                        with st.expander(expander_title):
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"Resolved ({resolved_count})", f"Unresolved ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("Date",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.info("No resolved items in this category.")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(category_df[category_df['Resulting Actions'].astype(str).isin(
                                        ['None', '', 'nan'])][
                                                     ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                                 hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("Date",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.success("All incidents in this category have been resolved.")

                st.markdown("---")
                st.markdown("#### General Incidents Group (Code starting with G)")
                df_general = df[df['รหัส'].str.startswith('G', na=False)].copy()
                if df_general.empty:
                    st.info("No General incident data found.")
                else:
                    general_categories = sorted([cat for cat in df_general['หมวด'].unique() if cat and pd.notna(cat)])
                    for category in general_categories:
                        category_df = df_general[df_general['หมวด'] == category]
                        total_count = len(category_df);
                        resolved_df = category_df[
                            ~category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])];
                        resolved_count = len(resolved_df);
                        unresolved_count = total_count - resolved_count
                        expander_title = f"{category} (Total: {total_count} | Resolved: {resolved_count} | Unresolved: {unresolved_count})"
                        with st.expander(expander_title):
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"Resolved ({resolved_count})", f"Unresolved ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("Date",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.info("No resolved items in this category.")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(category_df[category_df['Resulting Actions'].astype(str).isin(
                                        ['None', '', 'nan'])][
                                                     ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                                 hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("Date",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.success("All incidents in this category have been resolved.")

            with tab_summary:
                st.subheader("Severe Incident Resolution Summary (E-I & 3-5)")
                total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
                total_severe_psg9_incidents = metrics_data.get("total_severe_psg9_incidents", 0)
                total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", 0)
                total_severe_unresolved_psg9_incidents_val = metrics_data.get(
                    "total_severe_unresolved_psg9_incidents_val", 0)

                val_row3_total_pct = (
                            total_severe_unresolved_incidents_val / total_severe_incidents * 100) if isinstance(
                    total_severe_unresolved_incidents_val, int) and total_severe_incidents > 0 else 0
                val_row3_psg9_pct = (
                            total_severe_unresolved_psg9_incidents_val / total_severe_psg9_incidents * 100) if isinstance(
                    total_severe_unresolved_psg9_incidents_val, int) and total_severe_psg9_incidents > 0 else 0

                summary_action_data = [
                    {"Metric": "1. Number of Severe Incidents (E-I & 3-5)",
                     "All Incidents": f"{total_severe_incidents:,}", "PSG9 Only": f"{total_severe_psg9_incidents:,}"},
                    {"Metric": "2. Unresolved Severe Incidents (E-I & 3-5)",
                     "All Incidents": f"{total_severe_unresolved_incidents_val:,}" if isinstance(
                         total_severe_unresolved_incidents_val, int) else "N/A",
                     "PSG9 Only": f"{total_severe_unresolved_psg9_incidents_val:,}" if isinstance(
                         total_severe_unresolved_psg9_incidents_val, int) else "N/A"},
                    {"Metric": "3. % of Unresolved Severe Incidents", "All Incidents": f"{val_row3_total_pct:.2f}%",
                     "PSG9 Only": f"{val_row3_psg9_pct:.2f}%"}
                ]
                st.dataframe(pd.DataFrame(summary_action_data).set_index('Metric'), use_container_width=True)

    elif selected_analysis == "Persistence Risk Index":
        st.markdown("<h4 style='color: #001f3f;'>Persistence Risk Index</h4>", unsafe_allow_html=True)
        st.info(
            "This table scores incidents that occur repeatedly and have a high average severity, indicating chronic issues that may require systemic review.")
        persistence_df = calculate_persistence_risk_score(df, total_month)
        if not persistence_df.empty:
            display_df_persistence = persistence_df.rename(columns={
                'รหัส': 'Code',
                'ชื่ออุบัติการณ์ความเสี่ยง': 'Incident Name',
                'Persistence_Risk_Score': 'Persistence Index',
                'Average_Ordinal_Risk_Score': 'Avg Risk Score',
                'Incident_Rate_Per_Month': 'Rate (per month)',
                'Total_Occurrences': 'Total Count'
            })
            st.dataframe(
                display_df_persistence[
                    ['Code', 'Incident Name', 'Avg Risk Score', 'Persistence Index', 'Rate (per month)',
                     'Total Count']],
                use_container_width=True, hide_index=True,
                column_config={
                    "Avg Risk Score": st.column_config.NumberColumn(format="%.2f"),
                    "Rate (per month)": st.column_config.NumberColumn(format="%.2f"),
                    "Persistence Index": st.column_config.ProgressColumn(
                        "Persistence Risk Index",
                        help="Calculated from frequency and average severity. Higher value indicates a more chronic problem.",
                        min_value=0, max_value=2, format="%.2f"
                    )
                }
            )
            st.markdown("---")
            st.markdown("##### Analysis Graph of Chronic Problems")
            fig = px.scatter(
                persistence_df, x="Average_Ordinal_Risk_Score", y="Incident_Rate_Per_Month", size="Total_Occurrences",
                color="Persistence_Risk_Score", hover_name="ชื่ออุบัติการณ์ความเสี่ยง",
                color_continuous_scale=px.colors.sequential.Reds,
                size_max=60,
                labels={
                    "Average_Ordinal_Risk_Score": "Average Risk Score (Higher = More Severe)",
                    "Incident_Rate_Per_Month": "Incidents per Month (Higher = More Frequent)",
                    "Persistence_Risk_Score": "Persistence Index"
                },
                title="Distribution of Chronic Problems: Frequency vs. Severity"
            )
            fig.update_layout(xaxis_title="Average Severity", yaxis_title="Average Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data to analyze persistence risk.")

    elif selected_analysis == "Frequency Trend (Poisson)":
        st.markdown("<h4 style='color: #001f3f;'>Frequency Trend Index (Poisson)</h4>", unsafe_allow_html=True)
        with st.expander("Click to see explanation of calculation and interpretation"):
            st.markdown("""
                This table analyzes the **'frequency'** of incident occurrences over time using Poisson Regression, a statistical model for count data.
                - **Poisson Slope:** A technical value from the model on a logarithmic scale. A positive value means an increasing trend, negative means a decreasing trend.
                - **Change Rate (factor/month):** **(The most important column for interpretation)** This value is converted from the Slope (`e^Slope`) for easy reading.
                    - **Value > 1:** The frequency trend is increasing. For example, a value of `2.0` means the frequency is trending to double **(2x)** each month.
                    - **Value = 1:** There is no change in the trend.
                    - **Value < 1:** The frequency trend is decreasing. For example, a value of `0.8` means the frequency is trending to decrease to **80%** (a 20% reduction) each month.
            """)
        poisson_trend_df = calculate_frequency_trend_poisson(df)
        if not poisson_trend_df.empty:
            poisson_trend_df['Monthly_Change_Factor'] = np.exp(poisson_trend_df['Poisson_Trend_Slope'])
            display_df = poisson_trend_df.rename(columns={
                'รหัส': 'Code',
                'ชื่ออุบัติการณ์ความเสี่ยง': 'Incident Name',
                'Poisson_Trend_Slope': 'Frequency Trend (Slope)',
                'Total_Occurrences': 'Total Occurrences',
                'Months_Observed': 'Months Observed',
                'Monthly_Change_Factor': 'Change Rate (factor/month)'
            })
            display_cols_order = ['Code', 'Incident Name', 'Frequency Trend (Slope)', 'Change Rate (factor/month)',
                                  'Total Occurrences', 'Months Observed']
            st.dataframe(
                display_df[display_cols_order], use_container_width=True, hide_index=True,
                column_config={
                    "Change Rate (factor/month)": st.column_config.NumberColumn("Change Rate (factor/month)",
                                                                                format="%.2f"),
                    "Frequency Trend (Slope)": st.column_config.ProgressColumn(
                        "Frequency Trend (Higher = More Frequent)",
                        format="%.4f", min_value=display_df['Frequency Trend (Slope)'].min(),
                        max_value=display_df['Frequency Trend (Slope)'].max()),
                }
            )
            st.markdown("---")
            st.markdown("##### Drill-down per Incident: Distribution and Trend")
            options_list = display_df['Code'] + " | " + display_df['Incident Name'].fillna('')
            incident_to_plot = st.selectbox(
                'Select an incident to view its distribution graph:',
                options=options_list, index=0, key="sb_poisson_trend_final"
            )
            if incident_to_plot:
                selected_code = incident_to_plot.split(' | ')[0]
                fig = create_poisson_trend_plot(df, selected_code, display_df)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not calculate frequency trend. Data may be insufficient.")

    elif selected_analysis == "Executive Summary":
        df = st.session_state.get('processed_df', pd.DataFrame())
        metrics_data = st.session_state.get('metrics_data', {})
        df_freq = st.session_state.get('df_freq_for_display', pd.DataFrame())
        if df.empty:
            st.warning("No data found to generate the report. Please upload a new file.")
            st.stop()

        total_processed_incidents = metrics_data.get("total_processed_incidents", 0)
        total_sentinel_incidents_for_metric1 = metrics_data.get("total_sentinel_incidents_for_metric1", 0)
        total_psg9_incidents_for_metric1 = metrics_data.get("total_psg9_incidents_for_metric1", 0)
        total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
        total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", "N/A")
        total_month = metrics_data.get("total_month", 1)

        st.markdown("<h4 style='color: #001f3f;'>Executive Summary</h4>", unsafe_allow_html=True)
        if 'Occurrence Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Occurrence Date']) and df[
            'Occurrence Date'].notna().any():
            min_date_str = df['Occurrence Date'].min().strftime('%d/%m/%Y')
            max_date_str = df['Occurrence Date'].max().strftime('%d/%m/%Y')
            analysis_period_str = f"from {min_date_str} to {max_date_str} (total {total_month} months)"
        else:
            analysis_period_str = f"period not identifiable (total {total_month} months)"
        st.markdown(f"**Subject:** Hospital Incident Summary Report")
        st.markdown(f"**Analysis Period:** {analysis_period_str}")
        st.markdown(f"**Total Incidents Found:** {total_processed_incidents:,} items")
        st.markdown("---")

        st.subheader("1. Summary Dashboard")
        col1_m, col2_m, col3_m, col4_m, col5_m = st.columns(5)
        with col1_m:
            st.metric("Total Incidents", f"{total_processed_incidents:,}")
        with col2_m:
            st.metric("Sentinel Events", f"{total_sentinel_incidents_for_metric1:,}")
        with col3_m:
            st.metric("PSG9 Incidents", f"{total_psg9_incidents_for_metric1:,}")
        with col4_m:
            st.metric("High Severity (E-I & 3-5)", f"{total_severe_incidents:,}")
        with col5_m:
            label_m5 = "High Severity & Unresolved"
            value_m5 = f"{total_severe_unresolved_incidents_val:,}" if isinstance(total_severe_unresolved_incidents_val,
                                                                                  int) else "N/A"
            st.metric(label_m5, value_m5)
        st.markdown("---")

        st.subheader("2. Risk Matrix & Top 10 Incidents")
        col_matrix, col_top10 = st.columns(2)
        with col_matrix:
            st.markdown("##### Risk Matrix")
            matrix_data = pd.crosstab(df['Impact Level'], df['Frequency Level'])
            impact_order = ['5', '4', '3', '2', '1'];
            freq_order = ['1', '2', '3', '4', '5']
            matrix_data = matrix_data.reindex(index=impact_order, columns=freq_order, fill_value=0)
            impact_labels = {'5': "5 (Extreme)", '4': "4 (Major)", '3': "3 (Moderate)", '2': "2 (Minor)",
                             '1': "1 (Insignificant)"}
            freq_labels = {'1': "F1", '2': "F2", '3': "F3", '4': "F4", '5': "F5"}
            matrix_data_display = matrix_data.rename(index=impact_labels, columns=freq_labels)
            st.table(matrix_data_display)
        with col_top10:
            st.markdown("##### Top 10 Incidents (by Frequency)")
            if not df_freq.empty:
                df_freq_top10 = df_freq.nlargest(10, 'count').copy()
                display_top10 = df_freq_top10[['Incident', 'count']].rename(
                    columns={'Incident': 'Incident Code', 'count': 'Count'}).set_index('Incident Code')
                st.table(display_top10)
            else:
                st.warning("Cannot display Top 10.")
        st.markdown("---")

        st.subheader("3. Sentinel Events List")
        if sentinel_composite_keys:
            df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                str).str.strip()
            sent_rec_found = df[df['Sentinel code for check'].isin(sentinel_composite_keys)]
            if not sent_rec_found.empty:
                exec_sentinel_cols = ['Occurrence Date', 'Impact', 'รายละเอียดการเกิด', 'Resulting Actions']
                cols_to_display = [col for col in exec_sentinel_cols if col in sent_rec_found.columns]
                st.dataframe(sent_rec_found[cols_to_display].sort_values(by='Occurrence Date', ascending=False),
                             hide_index=True, use_container_width=True,
                             column_config={
                                 "Occurrence Date": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY",
                                                                                    width="medium"),
                                 "Impact": st.column_config.Column("Level", width="small"),
                                 "รายละเอียดการเกิด": st.column_config.Column("Description", width="large"),
                                 "Resulting Actions": st.column_config.Column("Resolution", width="large")
                             })
            else:
                st.info("No Sentinel Events found in this data period.")
        else:
            st.warning("Cannot analyze Sentinel Events.")
        st.markdown("---")

        st.subheader("4. Analysis by 9 Patient Safety Goals (PSG9)")
        psg9_summary_table_combined = create_psg9_summary_table(df)
        if psg9_summary_table_combined is not None and not psg9_summary_table_combined.empty:
            st.table(psg9_summary_table_combined)
        else:
            st.info("No incident data related to the 9 Patient Safety Goals found.")
        st.markdown("---")

        st.subheader("5. List of Unresolved Severe Incidents (E-I & 3-5)")
        if 'Resulting Actions' in df.columns:
            severe_conditions = df['Impact Level'].isin(['3', '4', '5'])
            unresolved_conditions = df['Resulting Actions'].astype(str).isin(['None', ''])
            df_severe_unresolved_exec = df[severe_conditions & unresolved_conditions]
            if not df_severe_unresolved_exec.empty:
                st.write(f"Found {df_severe_unresolved_exec.shape[0]} unresolved severe incidents:")
                st.dataframe(
                    df_severe_unresolved_exec[display_cols_common].sort_values(by='Occurrence Date', ascending=False),
                    hide_index=True, use_container_width=True,
                    column_config={"Occurrence Date": st.column_config.DatetimeColumn("Date", format="DD/MM/YYYY")})
            else:
                st.info("No unresolved severe incidents found in this data period.")
        else:
            st.warning("Column 'Resulting Actions' not found. Cannot display list of unresolved incidents.")
        st.markdown("---")

        st.subheader("6. Summary by Safety Goals")
        goal_definitions_exec = {
            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
            "Personnel Safety": "P:Personnel Safety Goals",
            "Organization Safety": "O:Organization Safety Goals"
        }
        for display_name, cat_name in goal_definitions_exec.items():
            st.markdown(f"##### {display_name}")
            is_org_safety_flag_combined = (display_name == "Organization Safety")
            e_up_non_numeric = [] if is_org_safety_flag_combined else ['A', 'B', 'C', 'D']
            e_up_numeric = ['1', '2'] if is_org_safety_flag_combined else None
            summary_table_goals = create_goal_summary_table(df, cat_name, e_up_non_numeric, e_up_numeric,
                                                            is_org_safety_table=is_org_safety_flag_combined)
            if summary_table_goals is not None and not summary_table_goals.empty:
                st.table(summary_table_goals)
            else:
                st.info(f"No data available for '{display_name}'")
        st.markdown("---")

        st.subheader("7. Top 5 Incidents with Increasing Frequency Trend")
        st.write(
            "Showing Top 5 incidents where the frequency of occurrence is trending upwards the fastest (from Poisson Regression).")
        poisson_trend_df_exec = calculate_frequency_trend_poisson(df)
        if not poisson_trend_df_exec.empty:
            top_freq_trending = poisson_trend_df_exec[poisson_trend_df_exec['Poisson_Trend_Slope'] > 0].head(5).copy()
            if not top_freq_trending.empty:
                top_freq_trending['Change Rate (factor/month)'] = np.exp(top_freq_trending['Poisson_Trend_Slope'])
                display_df_freq_trend = top_freq_trending.rename(
                    columns={'รหัส': 'Code', 'Poisson_Trend_Slope': 'Trend (Slope)', 'Total_Occurrences': 'Total Count',
                             'ชื่ออุบัติการณ์ความเสี่ยง': 'Incident Name'})
                display_freq_trend_table = display_df_freq_trend[
                    ['Code', 'Incident Name', 'Trend (Slope)', 'Change Rate (factor/month)', 'Total Count']].set_index(
                    'Code')
                st.table(display_freq_trend_table)
            else:
                st.success("✔️ No incidents with an increasing frequency trend were found in this period.")
        else:
            st.info("Insufficient data to analyze frequency trends.")
        st.markdown("---")

        st.subheader("8. Top 5 Chronic Incidents (Persistence Risk)")
        st.write(
            "Showing Top 5 incidents that occur frequently and have a high average severity, which may require systemic review.")
        persistence_df_exec = calculate_persistence_risk_score(df, total_month)
        if not persistence_df_exec.empty:
            top_persistence_incidents = persistence_df_exec.head(5)
            display_df_persistence = top_persistence_incidents.rename(
                columns={'รหัส': 'Code', 'Persistence_Risk_Score': 'Persistence Index',
                         'Average_Ordinal_Risk_Score': 'Avg Risk Score', 'ชื่ออุบัติการณ์ความเสี่ยง': 'Incident Name'})
            display_persistence_table = display_df_persistence[
                ['Code', 'Incident Name', 'Avg Risk Score', 'Persistence Index']].set_index('Code')
            st.table(display_persistence_table)
        else:
            st.info("Insufficient data to analyze persistence risk.")

    elif selected_analysis == "Chat with AI Assistant":
        st.markdown("<h4 style='color: #001f3f;'>AI Assistant</h4>", unsafe_allow_html=True)
        st.info(
            "Ask questions like 'Which incident is most severe?', 'Analyze code CPP101', 'What is the most chronic problem?', 'What has the highest increasing trend?'")

        try:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            AI_IS_CONFIGURED = True
        except Exception as e:
            st.error(f"⚠️ Could not configure AI. Please check your .streamlit/secrets.toml file and API Key: {e}")
            AI_IS_CONFIGURED = False

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        chat_history_container = st.container(height=400, border=True)
        with chat_history_container:
            for message in st.session_state.chat_messages:
                avatar = LOGO_URL if message["role"] == "assistant" else "❓"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask about risk data, or type an incident code to analyze..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="❓"):
                st.markdown(prompt)

            if AI_IS_CONFIGURED:
                with st.spinner("Thinking and analyzing for a moment..."):
                    df_for_ai = st.session_state.get('processed_df', pd.DataFrame())
                    total_months = st.session_state.get('metrics_data', {}).get('total_month', 1)
                    prompt_lower = prompt.lower()
                    final_ai_response_text = ""

                    if df_for_ai.empty:
                        final_ai_response_text = "I'm sorry, please upload a data file first before asking questions."
                    else:
                        incident_code_match = re.search(r'[A-Z]{3}\d{3}', prompt.upper())

                        if incident_code_match:
                            extracted_code = incident_code_match.group(0)
                            incident_df = df_for_ai[df_for_ai['รหัส'] == extracted_code]
                            if incident_df.empty:
                                final_ai_response_text = f"I'm sorry, I couldn't find any data for the code '{extracted_code}' in the file."
                            else:
                                incident_name = incident_df['ชื่ออุบัติการณ์ความเสี่ยง'].iloc[
                                    0] if 'ชื่ออุบัติการณ์ความเสี่ยง' in incident_df.columns and not incident_df[
                                    'ชื่ออุบัติการณ์ความเสี่ยง'].isnull().all() else "N/A"
                                total_count = len(incident_df)
                                severity_dist = incident_df['Impact'].value_counts().to_dict()
                                severity_text = ", ".join(
                                    [f"Level {k}: {v} times" for k, v in severity_dist.items()]) or "No data"
                                poisson_df = calculate_frequency_trend_poisson(df_for_ai)
                                trend_row = poisson_df[poisson_df['รหัส'] == extracted_code]
                                trend_text = f"Slope: {trend_row['Poisson_Trend_Slope'].iloc[0]:.4f}" if not trend_row.empty else "Insufficient data to calculate"
                                persistence_df = calculate_persistence_risk_score(df_for_ai, total_months)
                                persistence_row = persistence_df[persistence_df['รหัส'] == extracted_code]
                                persistence_text = f"Score: {persistence_row['Persistence_Risk_Score'].iloc[0]:.2f}" if not persistence_row.empty else "Insufficient data to calculate"
                                final_ai_response_text = f"""**Summary Report for Code: {extracted_code}**
- **Full Name:** {incident_name}
- **Total Occurrences:** {total_count} times
- **Severity Distribution:** {severity_text}
- **Occurrence Trend (Poisson):** {trend_text}
- **Persistence Risk Score:** {persistence_text}
"""
                        elif 'most frequent' in prompt_lower or 'highest frequency' in prompt_lower:
                            if 'ชื่ออุบัติการณ์ความเสี่ยง' not in df_for_ai.columns or df_for_ai[
                                'ชื่ออุบัติการณ์ความเสี่ยง'].isnull().all():
                                final_ai_response_text = "I'm sorry, I cannot analyze this as the 'Incident Name' data is missing."
                            else:
                                most_frequent_name = df_for_ai['ชื่ออุบัติการณ์ความเสี่ยง'].value_counts().idxmax()
                                count = df_for_ai['ชื่ออุบัติการณ์ความเสี่ยง'].value_counts().max()
                                incident_code = \
                                df_for_ai[df_for_ai['ชื่ออุบัติการณ์ความเสี่ยง'] == most_frequent_name]['รหัส'].iloc[0]
                                final_ai_response_text = f"The most frequent incident in the data is:\n- **Incident Name:** {most_frequent_name}\n- **Code:** {incident_code}\n- **Count:** {count} times."

                        elif 'most severe' in prompt_lower or 'most risky' in prompt_lower:
                            if 'Ordinal_Risk_Score' not in df_for_ai.columns:
                                risk_level_map_to_score = {"51": 21, "52": 22, "53": 23, "54": 24, "55": 25, "41": 16,
                                                           "42": 17, "43": 18, "44": 19, "45": 20, "31": 11, "32": 12,
                                                           "33": 13, "34": 14, "35": 15, "21": 6, "22": 7, "23": 8,
                                                           "24": 9, "25": 10, "11": 1, "12": 2, "13": 3, "14": 4,
                                                           "15": 5}
                                df_for_ai['Ordinal_Risk_Score'] = df_for_ai['Risk Level'].astype(str).map(
                                    risk_level_map_to_score)
                            if df_for_ai['Ordinal_Risk_Score'].isnull().all():
                                final_ai_response_text = "I'm sorry, I cannot analyze this as there is no risk score data."
                            else:
                                most_severe_incident = df_for_ai.loc[df_for_ai['Ordinal_Risk_Score'].idxmax()]
                                incident_details = most_severe_incident.get('รายละเอียดการเกิด',
                                                                            'No additional description.')
                                final_ai_response_text = f"""The most severe incident found is:
- **Incident Name:** {most_severe_incident.get('ชื่ออุบัติการณ์ความเสี่ยง', 'N/A')}
- **Code:** {most_severe_incident.get('รหัส', 'N/A')}
- **Severity Level:** {most_severe_incident.get('Impact', 'N/A')}
- **Date:** {most_severe_incident.get('Occurrence Date').strftime('%d/%m/%Y') if pd.notna(most_severe_incident.get('Occurrence Date')) else 'N/A'}
- **Description:** {incident_details}
"""
                        elif 'chronic' in prompt_lower or 'persistence' in prompt_lower:
                            persistence_df = calculate_persistence_risk_score(df_for_ai, total_months)
                            if persistence_df.empty:
                                final_ai_response_text = "I'm sorry, I could not analyze chronic problems, likely due to insufficient data."
                            else:
                                most_persistent = persistence_df.iloc[0]
                                incident_name = most_persistent.get('ชื่ออุบัติการณ์ความเสี่ยง', 'N/A')
                                incident_code = most_persistent.get('รหัส', 'N/A')
                                score = most_persistent.get('Persistence_Risk_Score', 0)
                                final_ai_response_text = f"""The most chronic problem (highest Persistence Risk) is:
- **Incident Name:** {incident_name}
- **Code:** {incident_code}
- **Persistence Score:** {score:.2f} (Higher is more chronic)
"""
                        elif 'highest trend' in prompt_lower or 'increasing trend' in prompt_lower:
                            poisson_df = calculate_frequency_trend_poisson(df_for_ai)
                            trending_up_df = poisson_df[poisson_df['Poisson_Trend_Slope'] > 0]
                            if trending_up_df.empty:
                                final_ai_response_text = "✔️ Good news! Based on the analysis, no incidents have a significantly increasing frequency trend."
                            else:
                                highest_trend = trending_up_df.iloc[0]
                                incident_name = highest_trend.get('ชื่ออุบัติการณ์ความเสี่ยง', 'N/A')
                                incident_code = highest_trend.get('รหัส', 'N/A')
                                slope = highest_trend.get('Poisson_Trend_Slope', 0)
                                change_factor = np.exp(slope)
                                final_ai_response_text = f"""The incident with the highest increasing frequency trend is:
- **Incident Name:** {incident_name}
- **Code:** {incident_code}
- **Change Rate:** The frequency is trending to increase by a factor of **{change_factor:.2f}x** each month.
"""
                        else:
                            final_ai_response_text = "I'm sorry, I don't understand the question. Please try using more specific keywords like 'code...', 'most frequent', 'most severe', 'chronic', or 'highest trend'."

                    with st.chat_message("assistant", avatar=LOGO_URL):
                        st.markdown(final_ai_response_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": final_ai_response_text})
            else:
                with st.chat_message("assistant", avatar=LOGO_URL):
                    st.error("Could not connect to the AI Assistant. Please check the API Key configuration.")
