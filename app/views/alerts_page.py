import streamlit as st
import pandas as pd
from app.utils.api_client import APIClient

def show_alerts():
    st.markdown('<div class="page-title">Epidemiological Alerts (from year 2021 onwards)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time alerts of emerging hotspots and regional outbreak threats</div>', unsafe_allow_html=True)

    with st.spinner("Fetching active alerts..."):
        res = APIClient.get("/alerts")

    if res and "alerts" in res:
        alerts_list = res["alerts"]
        if alerts_list:
            df_alerts = pd.DataFrame(alerts_list)
            
            # Count severity
            critical_cnt = len(df_alerts[df_alerts["risk_level"] == "CRITICAL"])
            high_cnt = len(df_alerts[df_alerts["risk_level"] == "HIGH"])
            moderate_cnt = len(df_alerts[df_alerts["risk_level"] == "MODERATE"])
            
            # Render visual alert counts
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                    <div class="metric-card metric-red">
                        <div class="metric-label">🔴 Critical Warnings</div>
                        <div class="metric-value">{critical_cnt}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class="metric-card metric-orange">
                        <div class="metric-label">🟠 High Risk Alerts</div>
                        <div class="metric-value">{high_cnt}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                    <div class="metric-card metric-teal">
                        <div class="metric-label">🟡 Moderate Risks</div>
                        <div class="metric-value">{moderate_cnt}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # High Risk Alerts & Emerging Outbreaks List
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Active Warnings & Interventions</div>', unsafe_allow_html=True)

            for _, row in df_alerts.iterrows():
                level = row["risk_level"]
                if level == "CRITICAL":
                    st.error(f"🚨 **CRITICAL RISK** in **{row['district']}, {row['state_ut']}** — {row['message']} (Predicted Cases: {row['cases']})")
                elif level == "HIGH":
                    st.warning(f"⚠️ **HIGH RISK** in **{row['district']}, {row['state_ut']}** — {row['message']} (Predicted Cases: {row['cases']})")
                else:
                    st.info(f"ℹ️ **MODERATE RISK** in **{row['district']}, {row['state_ut']}** — {row['message']} (Predicted Cases: {row['cases']})")
                    
            st.markdown('</div>', unsafe_allow_html=True)

            # Risk Ranking Table
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Outbreak Risk Rankings</div>', unsafe_allow_html=True)
            
            # Display sorted alerts table
            display_df = df_alerts[["state_ut", "district", "disease", "cases", "risk_level"]].copy()
            display_df.rename(columns={
                "state_ut": "State/UT",
                "district": "District",
                "disease": "Disease",
                "cases": "Recent Cases",
                "risk_level": "Risk Severity"
            }, inplace=True)
            
            # Sort critical/high first
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
            display_df["sort_order"] = display_df["Risk Severity"].map(severity_order)
            display_df = display_df.sort_values(by="sort_order").drop(columns="sort_order")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.success("🎉 All clear! No emerging disease outbreaks or critical alerts detected in recent records.")
    else:
        st.warning("Failed to load alerts data from backend API.")
