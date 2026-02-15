# import streamlit as st

# st.set_page_config(page_title="Drishti-AX Pulse", layout="wide")

# import pandas as pd
# import json
# import os
# import plotly.express as px

# @st.cache_data
# def load_and_aggregate_data(data_dir):
#     all_data = []
#     if not os.path.exists(data_dir):
#         st.error(f"Directory {data_dir} not found!")
#         return pd.DataFrame()

#     for file in os.listdir(data_dir):
#         if file.endswith(".json"):
#             with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
#                 try:
#                     d = json.load(f)
#                     # Extracting the parameters we selected
#                     all_data.append({
#                         "URL": d.get("metadata", {}).get("target_url"),
#                         "Sector": d.get("metadata", {}).get("category", "General"),
#                         "Score": d.get("metadata", {}).get("drishti_score", 0),
#                         "Tech": d.get("metadata", {}).get("tech_stack", "Unknown"),
#                         "Complexity": d.get("deep_scan", {}).get("sensory_cognitive", {}).get("reading_complexity_score", 0),
#                         "Violations": d.get("accessibility", {}).get("violations_count", 0),
#                         "JS_Errors": d.get("stability", {}).get("js_errors", 0)
#                     })
#                 except Exception as e:
#                     continue # Skip corrupt files
#     return pd.DataFrame(all_data)

# # --- LOAD DATA ---
# DATA_PATH = "reports/data"
# df = load_and_aggregate_data(DATA_PATH)

# if df.empty:
#     st.warning("No data found. Ensure your JSON files are in 'reports/data'.")
# else:
#     # --- DASHBOARD HEADER ---
#     st.title("🛡️ Drishti-AX: Visual Intelligence Pulse")
#     st.markdown(f"**Analyzing {len(df)} High-Impact Targets**")

#     # 1. Top Level Metrics (KPIs)
#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Avg Drishti Score", f"{df['Score'].mean():.1f}")
#     m2.metric("Total Violations", df['Violations'].sum())
#     m3.metric("Critical Stability Faults", df['JS_Errors'].sum())
#     m4.metric("Avg Cognitive Load", f"{df['Complexity'].mean():.1f}")

#     # --- PHASE 2 ANALYTICS ---
#     col_a, col_b = st.columns(2)

#     with col_a:
#         st.subheader("Sectoral Health Comparison")
#         # Compare sectors to see who is failing the disabled community most
#         sector_chart = px.bar(
#             df.groupby("Sector")["Score"].mean().sort_values().reset_index(),
#             x="Score", y="Sector", orientation='h',
#             color="Score", color_continuous_scale="RdYlGn",
#             title="Accessibility Score by Industry"
#         )
#         st.plotly_chart(sector_chart, use_container_width=True)

#     with col_b:
#         st.subheader("Stability vs. Score Correlation")
#         # Show how JS errors destroy the score
#         scatter = px.scatter(
#             df, x="JS_Errors", y="Score", size="Violations", 
#             color="Sector", hover_name="URL",
#             title="The Cost of Technical Instability"
#         )
#         st.plotly_chart(scatter, use_container_width=True)

#     # --- THE WALL OF SHAME ---
#     st.divider()
#     st.subheader("Bottom 10 Worst Performers")
#     shame_df = df.nsmallest(10, "Score")[["URL", "Sector", "Score", "Violations", "JS_Errors"]]
#     st.dataframe(shame_df, use_container_width=True)

#     # --- INDIVIDUAL LOOKUP ---
#     st.sidebar.header("Individual Audit")
#     search_url = st.sidebar.selectbox("Select Target URL", df["URL"].unique())
#     if search_url:
#         site_info = df[df["URL"] == search_url].iloc[0]
#         st.sidebar.info(f"""
#         **Tech Stack:** {site_info['Tech']}  
#         **Readability:** {site_info['Complexity']}  
#         **JS Stability:** {site_info['JS_Errors']} Errors
#         """)


# def suggest_fix(violation_id, failure_msg):
#     """Maps technical violations to actionable code fixes."""
#     msg = failure_msg.lower()
#     if "aria-allowed-role" in violation_id:
#         return "❌ Invalid ARIA role. Remove role='presentation' from focusable elements like buttons to ensure they are visible to screen readers."
#     if "button-name" in violation_id:
#         return "❌ Empty Button. Add an 'aria-label' attribute or visible text inside the <button> tag."
#     if "image-alt" in violation_id:
#         return "❌ Missing Alt Text. Add alt='description' to the <img> tag so non-visual users understand the content."
#     if "landmark" in msg:
#         return "❌ Layout Error. Navigation or main content is not inside a landmark. Wrap content in <main>, <nav>, or <header>."
#     return "Refactor this element to meet WCAG 2.1 AA success criteria."



# target = st.selectbox("Select Target for Remediation", df["URL"].unique())

# if target:
#     # Load the specific JSON file for this target
#     # Based on your example: report_agriculture_karnataka_gov_in.json
#     filename = f"report_{target.split('//')[-1].replace('.', '_').replace('/', '')}.json"
#     filepath = os.path.join("reports/data", filename)

#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as f:
#             full_data = json.load(f)
        
#         st.subheader(f"Detailed Violations for {target}")
        
#         # Loop through violations from your JSON structure
#         violations = full_data.get("accessibility", {}).get("violations", [])
        
#         for v in violations:
#             with st.expander(f"🔴 {v['id'].upper()} - Impact: {v['impact']}"):
#                 col1, col2 = st.columns([2, 1])
                
#                 # Get specific node data
#                 node = v.get('nodes', [{}])[0]
#                 patch, doc_link = suggest_fix(v['id'], node.get('failure_summary', ''))
                
#                 with col1:
#                     st.code(node['html'], language='html')
#                     st.warning(f"**Issue:** {node['failure_summary']}")
                
#                 with col2:
#                     st.success("**Suggested Remediation Patch:**")
#                     st.info(patch)
#                     st.markdown(f"[View Documentation]({doc_link})")
#     else:
#         st.error("JSON report file missing.")










import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# MUST BE FIRST
st.set_page_config(page_title="Drishti-AX Pulse", layout="wide")



# --- DATA LOADING ---
@st.cache_data
def load_and_aggregate_data(data_dir):
    all_data = []
    if not os.path.exists(data_dir):
        st.error(f"Directory {data_dir} not found!")
        return pd.DataFrame()

    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                try:
                    d = json.load(f)
                    all_data.append({
                        "URL": d.get("metadata", {}).get("target_url"),
                        "Sector": d.get("metadata", {}).get("category", "General"),
                        "Score": d.get("metadata", {}).get("drishti_score", 0),
                        "Tech": d.get("metadata", {}).get("tech_stack", "Unknown"),
                        "Complexity": d.get("deep_scan", {}).get("sensory_cognitive", {}).get("reading_complexity_score", 0),
                        "Violations": d.get("accessibility", {}).get("violations_count", 0),
                        "JS_Errors": d.get("stability", {}).get("js_errors", 0),
                        "filename": file # Essential for the remediation lookup
                    })
                except Exception:
                    continue 
    return pd.DataFrame(all_data)

# --- SUGGEST FIX FUNCTION (FIXED) ---
def suggest_fix(violation_id, failure_msg):
    """Maps technical violations to actionable code fixes and help links."""
    msg = failure_msg.lower()
    doc_link = "https://dequeuniversity.com/rules/axe/4.4/" + violation_id
    
    if "aria-allowed-role" in violation_id:
        patch = "❌ Invalid ARIA role. Remove role='presentation' from focusable elements like buttons."
    elif "button-name" in violation_id:
        patch = "❌ Empty Button. Add an 'aria-label' attribute or visible text inside the <button> tag."
    elif "image-alt" in violation_id:
        patch = "❌ Missing Alt Text. Add alt='description' to the <img> tag."
    elif "landmark" in msg:
        patch = "❌ Layout Error. Wrap content in <main>, <nav>, or <header> landmarks."
    else:
        patch = "Refactor this element to meet WCAG 2.1 AA success criteria."
    
    # RETURN TWO VALUES to satisfy 'patch, doc_link = suggest_fix(...)'
    return patch, doc_link


def get_global_violation_stats(data_dir, filenames):
    """Phase 2: Global priority engine. Calculates which fixes move the needle most."""
    stats = []
    impact_weights = {"critical": 10, "serious": 5, "moderate": 2, "minor": 1}
    
    for file in filenames:
        try:
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                d = json.load(f)
                violations = d.get("accessibility", {}).get("violations", [])
                for v in violations:
                    impact = v.get("impact", "minor")
                    stats.append({
                        "Violation ID": v.get("id"),
                        "Impact": impact,
                        "Weight": impact_weights.get(impact, 1)
                    })
        except:
            continue
            
    if not stats:
        return pd.DataFrame()
        
    v_df = pd.DataFrame(stats)
    # Aggregate: Count occurrences and sum the weights
    priority_df = v_df.groupby("Violation ID").agg(
        Occurrences=("Violation ID", "count"),
        Total_Weight=("Weight", "sum")
    ).reset_index()
    
    # The 'Priority Score' is the total impact across all 5,000 sites
    return priority_df.sort_values(by="Total_Weight", ascending=False).head(10)




def calculate_projected_score(current_avg, priorities, fixed_ids):
    """
    Simulates score improvement.
    Each major fix is estimated to recover 5-12 points depending on its global weight.
    """
    if not fixed_ids:
        return current_avg
    
    improvement = 0
    for fix_id in fixed_ids:
        # Find the weight of the violation in the global priority list
        match = priorities[priorities["Violation ID"] == fix_id]
        if not match.empty:
            # Logic: More frequent/severe issues provide higher recovery
            # Scaled to ensure we don't exceed a score of 100
            improvement += (match["Total_Weight"].values[0] / priorities["Total_Weight"].sum()) * 15
            
    return min(100.0, current_avg + improvement)




# --- RUN DASHBOARD ---
DATA_PATH = "reports/data"
df = load_and_aggregate_data(DATA_PATH)

if df.empty:
    st.warning("No data found. Ensure your JSON files are in 'reports/data'.")
else:
    st.title("🛡️ Drishti-AX: Visual Intelligence Pulse")
    
    # 1. KPIs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Score", f"{df['Score'].mean():.1f}")
    m2.metric("Total Violations", df['Violations'].sum())
    m3.metric("JS Faults", df['JS_Errors'].sum())
    m4.metric("Avg Complexity", f"{df['Complexity'].mean():.1f}")

    # 2. Charts
    col_a, col_b = st.columns(2)
    with col_a:
        sector_chart = px.bar(df.groupby("Sector")["Score"].mean().sort_values().reset_index(),
                             x="Score", y="Sector", orientation='h', color="Score", title="Health by Industry")
        st.plotly_chart(sector_chart, use_container_width=True)
    with col_b:
        scatter = px.scatter(df, x="JS_Errors", y="Score", size="Violations", color="Sector", hover_name="URL")
        st.plotly_chart(scatter, use_container_width=True)

    # 3. Wall of Shame
    # st.divider()
    # st.subheader("🚨 Bottom 10 Worst Performers")
    # st.dataframe(df.nsmallest(10, "Score")[["URL", "Sector", "Score", "Violations"]], use_container_width=True)


    st.divider()
    st.subheader("🍁 Bottom 10 Worst Performers")

# Sort by Score (Ascending) then by Violations (Descending)
    shame_df = df.sort_values(
    by=["Score", "Violations"], 
    ascending=[True, False]
    ).head(10)

# Display with a custom index to show ranking 1-10
    shame_df.index = range(1, len(shame_df) + 1)
    st.dataframe(
    shame_df[["URL", "Sector", "Score", "Violations", "JS_Errors"]], 
    use_container_width=True
    )

    

     # --- VIOLATION HEATMAP ---
    st.divider()
    st.subheader("🕸️ WCAG Filtered Failure Matrix ")
    
    # 1. Add the Filter Toggle
    # We default it to Critical and Serious to keep the heatmap readable
    impact_filter = st.multiselect(
        "Filter by Impact Level:",
        options=["critical", "serious", "moderate", "minor"],
        default=["critical", "serious"],
        help="Filter the matrix to focus on high-priority failures."
    )

    heatmap_list = []
    # Loop through the bottom 10 sites
    for _, row in shame_df.iterrows():
        try:
            with open(os.path.join(DATA_PATH, row['filename']), 'r', encoding='utf-8') as f:
                report = json.load(f)
                violations = report.get("accessibility", {}).get("violations", [])
                
                for v in violations:
                    v_impact = v.get("impact", "minor")
                    # ONLY add to heatmap if it matches the selected impact levels
                    if v_impact in impact_filter:
                        heatmap_list.append({
                            "Site": row['URL'].replace("https://", "").replace("www.", "")[:25], 
                            "Violation ID": v['id'],
                            "Impact": v_impact,
                            "Presence": 1
                        })
        except:
            continue

    if heatmap_list:
        h_df = pd.DataFrame(heatmap_list)
        # Pivot: Sites vs. Violation IDs
        pivot_df = h_df.pivot_table(index="Site", columns="Violation ID", values="Presence", fill_value=0)
        
        # 2. Render the Heatmap
        fig_heat = px.imshow(
            pivot_df,
            labels=dict(x="WCAG Rule", y="Website", color="Detected"),
            color_continuous_scale="Reds",
            aspect="auto",
            title=f"Distribution of {', '.join(impact_filter).title()} Failures"
        )
        
        fig_heat.update_xaxes(side="top", tickangle=-45)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.caption("💡 **Tip:** A solid vertical line indicates a systemic issue across your entire sector.")
    else:
        st.info("No violations match your selected filters. Try including 'Moderate' or 'Minor' impacts.")


    


    # --- SECTION 6: REMEDIATION PRIORITY (GLOBAL IMPACT) ---
    st.divider()
    st.header("🎯 Remediation Priority: Global Impact")
    st.markdown("Which single fix will improve the score of the entire 5,000-site dataset the most?")

    priority_df = get_global_violation_stats(DATA_PATH, df['filename'].tolist())

    if not priority_df.empty:
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            # Bar chart showing the "Weight" of each problem
            fig_priority = px.bar(
                priority_df, 
                x="Total_Weight", 
                y="Violation ID", 
                orientation='h',
                color="Total_Weight",
                color_continuous_scale="Viridis",
                title="Fix Priority (Weighted by Frequency & Severity)",
                labels={"Total_Weight": "Impact Score", "Violation ID": "WCAG Rule"}
            )
            st.plotly_chart(fig_priority, use_container_width=True)
            
        with col_p2:
            st.write("### 🍀 Top 3 Priority Fixes")
            for i, row in priority_df.head(3).iterrows():
                st.info(f"**{row['Violation ID'].upper()}**\n\nAffects {row['Occurrences']} sites. Fixing this is your highest ROI.")
                
        st.caption("Calculation: $Priority Score = \\sum (Occurrences \\times Impact Weight)$")
    else:
        st.info("Insufficient global data to calculate priorities.")


#  Individual Audit (Sidebar)
    st.sidebar.header("Individual Audit")
    search_url = st.sidebar.selectbox("Select Target URL", df["URL"].unique())
    if search_url:
        site_info = df[df["URL"] == search_url].iloc[0]
        st.sidebar.info(f"""
        **Tech Stack:** {site_info['Tech']}  
        **Readability:** {site_info['Complexity']}  
        **JS Stability:** {site_info['JS_Errors']} Errors
        """)

    # 4. REMEDIATION HUB (FIXED)
    st.divider()
    st.header(" Intelligent Remediation Engine")
    target = st.selectbox("Select Target for Remediation", df["URL"].unique())

    if target:
        # Pull the filename from our dataframe instead of guessing the string format
        target_filename = df[df["URL"] == target]["filename"].iloc[0]
        filepath = os.path.join(DATA_PATH, target_filename)

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            
            st.subheader(f"Detailed Violations for {target}")
            violations = full_data.get("accessibility", {}).get("violations", [])
            
            for v in violations:
                with st.expander(f"🔴 {v['id'].upper()} - Impact: {v.get('impact', 'unknown')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    # Safety check for nodes
                    nodes = v.get('nodes', [])
                    if nodes:
                        node = nodes[0]
                        # CALLING THE FIXED FUNCTION
                        patch, doc_link = suggest_fix(v['id'], node.get('failure_summary', ''))
                        
                    #     with col1:
                    #         st.code(node.get('html', 'N/A'), language='html')
                    #         st.warning(f"**Issue:** {node.get('failure_summary', 'No summary provided')}")
                        
                    #     with col2:
                    #         st.success("**Suggested Remediation Patch:**")
                    #         st.info(patch)
                    #         st.markdown(f"[View Documentation]({doc_link})")
                    # else:
                    #     st.write("No specific code snippet found for this violation.")

                        
                        with st.expander(f"🔴 {v['id'].upper()} - (Impact: {v.get('impact')})"):
                        # c1, c2 = st.columns([2, 1])
                         with col1:
                            st.markdown("**Broken Code Snippet:**")
                            st.code(node.get('html', 'N/A'), language='html')
                            st.warning(f"**Issue Details:** {node.get('failure_summary')}")
                         with col2:
                            st.success("**Suggested Patch:**")
                            st.info(patch)
                            st.markdown(f"[View WCAG Guidelines]({doc_link})")


        else:
            st.error(f"JSON report file missing: {target_filename}")

    


    # --- SECTION 7: THE REDEMPTION SIMULATOR ---
    st.divider()
    st.header("🚀 Redemption Simulator")
    st.markdown("Select which systemic issues you plan to fix to see the projected impact on the state's health.")

    if not priority_df.empty:
        c_sim1, c_sim2 = st.columns([1, 2])
        
        with c_sim1:
            st.write("### 🛠️ Select Fixes")
            to_fix = st.multiselect(
                "Select issues to remediate:",
                options=priority_df["Violation ID"].tolist(),
                help="These are the top 10 issues identified in the Priority Chart above."
            )
        
        current_avg = df["Score"].mean()
        projected_avg = calculate_projected_score(current_avg, priority_df, to_fix)
        
        with c_sim2:
            # Visualizing the delta
            st.write("### 📈 Projected Health Improvement")
            delta = projected_avg - current_avg
            
            st.metric(
                label="Projected Average Score", 
                value=f"{projected_avg:.2f}", 
                delta=f"+{delta:.2f}" if delta > 0 else None,
                delta_color="normal"
            )
            
            # A progress bar to visualize the 'Health' of the state
            st.progress(projected_avg / 100)
            
            if delta > 10:
                st.balloons()
                st.success(f"🔥 Critical Impact! Fixing these {len(to_fix)} items restores major accessibility to the region.")
            elif delta > 0:
                st.info("Gradual improvement detected. Every fix helps a citizen.")
    else:
        st.warning("Priority data unavailable for simulation.")








# import streamlit as st
# import pandas as pd
# import json
# import os
# import plotly.express as px

# # MUST BE THE FIRST STREAMLIT COMMAND
# st.set_page_config(page_title="Drishti-AX Pulse", layout="wide")

# # --- DATA LOADING ---
# @st.cache_data
# def load_and_aggregate_data(data_dir):
#     all_data = []
#     if not os.path.exists(data_dir):
#         st.error(f"Directory {data_dir} not found!")
#         return pd.DataFrame()

#     for file in os.listdir(data_dir):
#         if file.endswith(".json"):
#             with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
#                 try:
#                     d = json.load(f)
#                     all_data.append({
#                         "URL": d.get("metadata", {}).get("target_url"),
#                         "Sector": d.get("metadata", {}).get("category", "General"),
#                         "Score": d.get("metadata", {}).get("drishti_score", 0),
#                         "Tech": d.get("metadata", {}).get("tech_stack", "Unknown"),
#                         "Complexity": d.get("deep_scan", {}).get("sensory_cognitive", {}).get("reading_complexity_score", 0),
#                         "Violations": d.get("accessibility", {}).get("violations_count", 0),
#                         "JS_Errors": d.get("stability", {}).get("js_errors", 0),
#                         "filename": file 
#                     })
#                 except Exception:
#                     continue 
#     return pd.DataFrame(all_data)

# # --- SUGGEST FIX FUNCTION (FIXED UNPACKING) ---
# def suggest_fix(violation_id, failure_msg):
#     """Maps technical violations to actionable fixes and documentation."""
#     msg = failure_msg.lower()
#     # Dynamic documentation link based on Axe-core IDs
#     doc_link = f"https://dequeuniversity.com/rules/axe/4.4/{violation_id}"
    
#     if "aria-allowed-role" in violation_id:
#         patch = "❌ Invalid ARIA role. Remove role='presentation' from focusable elements like buttons."
#     elif "button-name" in violation_id:
#         patch = "❌ Empty Button. Add an 'aria-label' attribute or visible text inside the <button> tag."
#     elif "image-alt" in violation_id:
#         patch = "❌ Missing Alt Text. Add alt='description' to the <img> tag."
#     elif "landmark" in msg:
#         patch = "❌ Layout Error. Wrap content in <main>, <nav>, or <header> landmarks."
#     else:
#         patch = "Refactor this element to meet WCAG 2.1 AA success criteria."
    
#     return patch, doc_link

# # --- INITIALIZE DATA ---
# DATA_PATH = "reports/data"
# df = load_and_aggregate_data(DATA_PATH)

# if df.empty:
#     st.warning("No data found. Ensure your JSON files are in 'reports/data'.")
# else:
#     # --- SECTION 1: PULSE METRICS ---
#     st.title("🛡️ Drishti-AX: Visual Intelligence Pulse")
#     st.markdown(f"**Analyzing {len(df)} High-Impact Targets**")

#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Avg Score", f"{df['Score'].mean():.1f}")
#     m2.metric("Total Violations", df['Violations'].sum())
#     m3.metric("JS Stability Faults", df['JS_Errors'].sum())
#     m4.metric("Avg Cognitive Load", f"{df['Complexity'].mean():.1f}")

#     # --- SECTION 2: ANALYTICS CHARTS ---
#     col_a, col_b = st.columns(2)
#     with col_a:
#         st.subheader("Sectoral Health Comparison")
#         sector_chart = px.bar(
#             df.groupby("Sector")["Score"].mean().sort_values().reset_index(),
#             x="Score", y="Sector", orientation='h',
#             color="Score", color_continuous_scale="RdYlGn"
#         )
#         st.plotly_chart(sector_chart, use_container_width=True)

#     with col_b:
#         st.subheader("Stability vs. Score")
#         scatter = px.scatter(
#             df, x="JS_Errors", y="Score", size="Violations", 
#             color="Sector", hover_name="URL"
#         )
#         st.plotly_chart(scatter, use_container_width=True)

#     # --- SECTION 3: THE WALL OF SHAME ---
#     st.divider()
#     st.subheader("🚨 Bottom 10 Performers")
#     shame_df = df.nsmallest(10, "Score")[["URL", "Sector", "Score", "Violations"]]
#     st.dataframe(shame_df, use_container_width=True)

#     # --- SECTION 4: INDIVIDUAL LOOKUP (SIDEBAR) ---
#     st.sidebar.header("Site Forensic Summary")
#     search_url = st.sidebar.selectbox("Select Target URL", df["URL"].unique())
#     if search_url:
#         site_info = df[df["URL"] == search_url].iloc[0]
#         st.sidebar.info(f"""
#         **Tech Stack:** {site_info['Tech']}  
#         **Readability:** {site_info['Complexity']}  
#         **JS Errors:** {site_info['JS_Errors']}
#         """)

#     # --- SECTION 5: INTELLIGENT REMEDIATION ENGINE ---
#     st.divider()
#     st.header("🔍 Automated Remediation Patch Generator")
    
#     # Use the sidebar selection to drive the remediation view
#     if search_url:
#         target_file = df[df["URL"] == search_url]["filename"].iloc[0]
#         filepath = os.path.join(DATA_PATH, target_file)

#         if os.path.exists(filepath):
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 full_report = json.load(f)
            
#             st.subheader(f"Detailed Repair Guide for {search_url}")
#             violations = full_report.get("accessibility", {}).get("violations", [])
            
#             if not violations:
#                 st.success("Target is compliant.")
#             else:
#                 for v in violations:
#                     nodes = v.get('nodes', [])
#                     if nodes:
#                         node = nodes[0]
#                         # CALLING THE FIXED FUNCTION (Now returns 2 values)
#                         patch, doc_link = suggest_fix(v['id'], node.get('failure_summary', ''))
                        
#                         with st.expander(f"🔴 {v['id'].upper()} (Impact: {v.get('impact', 'unknown')})"):
#                             c1, c2 = st.columns([2, 1])
#                             with c1:
#                                 st.code(node.get('html', 'N/A'), language='html')
#                                 st.warning(f"**Issue:** {node.get('failure_summary', 'Check WCAG tags')}")
#                             with c2:
#                                 st.success("**Remediation Patch:**")
#                                 st.info(patch)
#                                 st.markdown(f"[View Guidelines]({doc_link})")
#         else:
#             st.error("JSON data missing for this target.")














# import streamlit as st
# import pandas as pd
# import json
# import os
# import plotly.express as px

# # MUST BE FIRST: Prevent circular import issues and set layout
# st.set_page_config(page_title="Drishti-AX Pulse & Remediation", layout="wide")

# # --- REMEDIATION LOGIC ---
# def suggest_fix(violation_id, failure_msg):
#     """Phase 2: Maps technical violations to actionable code fixes."""
#     msg = failure_msg.lower()
#     if "aria-allowed-role" in violation_id:
#         return "❌ Invalid ARIA role. Remove role='presentation' from focusable elements like buttons to ensure they are visible to screen readers."
#     if "button-name" in violation_id:
#         return "❌ Empty Button. Add an 'aria-label' attribute or visible text inside the <button> tag."
#     if "image-alt" in violation_id:
#         return "❌ Missing Alt Text. Add alt='description' to the <img> tag so non-visual users understand the content."
#     if "landmark" in msg:
#         return "❌ Layout Error. Navigation or main content is not inside a landmark. Wrap content in <main>, <nav>, or <header>."
#     return "Refactor this element to meet WCAG 2.1 AA success criteria."

# # --- DATA LOADING WITH FILENAME TRACKING ---
# @st.cache_data
# def load_and_aggregate_data(data_dir):
#     all_data = []
#     if not os.path.exists(data_dir):
#         st.error(f"Directory {data_dir} not found!")
#         return pd.DataFrame()

#     for file in os.listdir(data_dir):
#         if file.endswith(".json"):
#             with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
#                 try:
#                     d = json.load(f)
#                     # We add 'filename' so we can reopen the specific JSON for deep-dives
#                     all_data.append({
#                         "URL": d.get("metadata", {}).get("target_url"),
#                         "Sector": d.get("metadata", {}).get("category", "General"),
#                         "Score": d.get("metadata", {}).get("drishti_score", 0),
#                         "Tech": d.get("metadata", {}).get("tech_stack", "Unknown"),
#                         "Complexity": d.get("deep_scan", {}).get("sensory_cognitive", {}).get("reading_complexity_score", 0),
#                         "Violations": d.get("accessibility", {}).get("violations_count", 0),
#                         "JS_Errors": d.get("stability", {}).get("js_errors", 0),
#                         "filename": file 
#                     })
#                 except Exception:
#                     continue 
#     return pd.DataFrame(all_data)

# # --- LOAD DATA ---
# DATA_PATH = "reports/data"
# df = load_and_aggregate_data(DATA_PATH)

# if df.empty:
#     st.warning("No data found. Ensure your JSON files are in 'reports/data'.")
# else:
#     # --- DASHBOARD HEADER ---
#     st.title("🛡️ Drishti-AX: Visual Intelligence Pulse")
#     st.markdown(f"**Phase 2: Analyzing {len(df)} High-Impact Targets**")

#     # 1. Top Level Metrics (KPIs)
#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Avg Drishti Score", f"{df['Score'].mean():.1f}")
#     m2.metric("Total Violations", df['Violations'].sum())
#     m3.metric("Critical Stability Faults", df['JS_Errors'].sum())
#     m4.metric("Avg Cognitive Load", f"{df['Complexity'].mean():.1f}")

#     # --- ANALYTICS SECTION ---
#     col_a, col_b = st.columns(2)
#     with col_a:
#         st.subheader("Sectoral Health Comparison")
#         sector_chart = px.bar(
#             df.groupby("Sector")["Score"].mean().sort_values().reset_index(),
#             x="Score", y="Sector", orientation='h',
#             color="Score", color_continuous_scale="RdYlGn",
#             title="Accessibility Score by Industry"
#         )
#         st.plotly_chart(sector_chart, use_container_width=True)

#     with col_b:
#         st.subheader("Stability vs. Score Correlation")
#         scatter = px.scatter(
#             df, x="JS_Errors", y="Score", size="Violations", 
#             color="Sector", hover_name="URL",
#             title="The Cost of Technical Instability"
#         )
#         st.plotly_chart(scatter, use_container_width=True)

#     # --- THE WALL OF SHAME ---
#     st.divider()
#     st.subheader("🚨 Bottom 10 Worst Performers")
#     shame_df = df.nsmallest(10, "Score")[["URL", "Sector", "Score", "Violations", "JS_Errors"]]
#     st.dataframe(shame_df, use_container_width=True)

#     # --- INDIVIDUAL REMEDIATION HUB (Deep Dive) ---
#     st.divider()
#     st.header("🔍 Intelligent Remediation Engine")
#     search_url = st.selectbox("Select a target for automated repair suggestions:", df["URL"].unique())

#     if search_url:
#         target_row = df[df["URL"] == search_url].iloc[0]
        
#         # Open the specific JSON file for deep forensics
#         with open(os.path.join(DATA_PATH, target_row['filename']), 'r', encoding='utf-8') as f:
#             full_report = json.load(f)
        
#         # Display forensic overview
#         st.sidebar.header("Audit Summary")
#         st.sidebar.info(f"""
#         **Tech:** {target_row['Tech']}  
#         **Complexity:** {target_row['Complexity']}  
#         **JS Errors:** {target_row['JS_Errors']}
#         """)

#         violations = full_report.get("accessibility", {}).get("violations", [])
#         if not violations:
#             st.success("Target is fully compliant!")
#         else:
#             for v in violations:
#                 nodes = v.get("nodes", [])
#                 if nodes:
#                     node = nodes[0]
#                     with st.expander(f"🔴 {v.get('id', 'Issue').upper()} - Impact: {v.get('impact')}"):
#                         st.markdown(f"**Description:** {v.get('description')}")
#                         st.code(node.get("html", "N/A"), language="html")
                        
#                         # Generate the Phase 2 Patch
#                         patch = suggest_fix(v.get('id'), node.get('failure_summary', ''))
#                         st.success(f"**Remediation Patch:** {patch}")
#                         if v.get("helpUrl"):
#                             st.markdown(f"[View WCAG Guidelines]({v['helpUrl']})")