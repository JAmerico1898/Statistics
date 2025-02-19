import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Interactive Statistics Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        padding-top: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #283593;
    }
    .concept-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .example-box {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #FFF8E1;
        padding: 0.8rem;
        border-radius: 0.3rem;
        font-family: monospace;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>Interactive Statistics Learning</h1>", unsafe_allow_html=True)
st.markdown("""
This interactive app will help you understand key statistical concepts through visualizations and 
hands-on examples. Choose a topic from the sidebar to begin exploring.
""")

# Sidebar navigation
topic = st.sidebar.selectbox(
    "Select a Topic",
    ["Types of Variables", "Types of Data", "Measures of Central Tendency", "Measures of Dispersion"]
)

# Generate example datasets
@st.cache_data
def generate_example_data():
    # Continuous data
    continuous_data = np.random.normal(170, 10, 1000)  # Heights in cm
    
    # Categorical data
    categories = ['Red', 'Blue', 'Green', 'Yellow', 'Purple']
    categorical_data = np.random.choice(categories, 1000)
    
    # Ordinal data
    ordinal_levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
    ordinal_data = np.random.choice(ordinal_levels, 1000)
    
    # Time series data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    time_series = pd.Series(np.cumsum(np.random.randn(365) * 10 + 1), index=dates)
    
    # Cross-sectional data
    cross_sectional = pd.DataFrame({
        'Age': np.random.normal(35, 12, 100),
        'Income': np.random.normal(55000, 15000, 100),
        'Years_Education': np.random.normal(14, 3, 100),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    # Panel data
    panel_regions = ['North', 'South', 'East', 'West']
    panel_years = list(range(2010, 2024))
    region_data = []
    for region in panel_regions:
        base_gdp = np.random.randint(800, 1500)
        for year in panel_years:
            growth = np.random.normal(0.03, 0.02)
            gdp = base_gdp * (1 + growth) ** (year - 2010)
            unemployment = np.random.normal(5, 2)
            population = np.random.normal(1000000, 300000)
            region_data.append({
                'Region': region,
                'Year': year,
                'GDP_Billions': gdp,
                'Unemployment_Rate': unemployment,
                'Population': population
            })
    panel_data = pd.DataFrame(region_data)
    
    return {
        'continuous': continuous_data,
        'categorical': categorical_data,
        'ordinal': ordinal_data,
        'time_series': time_series,
        'cross_sectional': cross_sectional,
        'panel': panel_data
    }

example_data = generate_example_data()

# Topic 1: Types of Variables
if topic == "Types of Variables":
    st.markdown("<h2 class='section-header'>Types of Variables</h2>", unsafe_allow_html=True)
    
    variable_type = st.radio(
        "Select a variable type to explore:",
        ["Continuous", "Categorical", "Ordinal"]
    )
    
    if variable_type == "Continuous":
        st.markdown("<div class='concept-box'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-header'>Continuous Variables</h3>", unsafe_allow_html=True)
        st.markdown("""
        Continuous variables can take any value within a range. They are measured on a scale and can 
        be meaningfully divided into smaller increments, including fractional or decimal values.
        
        **Examples:** Height, weight, temperature, time, income, age
        
        **Key characteristics:**
        - Can take an infinite number of possible values within a range
        - Can be meaningfully subdivided into smaller parts
        - Mathematical operations (addition, subtraction, etc.) make sense
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interactive visualization for continuous data
        st.markdown("<h4>Interactive Exploration: Height Distribution</h4>", unsafe_allow_html=True)
        
        # Allow students to adjust the visualization
        bin_count = st.slider("Number of histogram bins:", min_value=5, max_value=50, value=20)
        height_unit = st.selectbox("Select unit:", ["Centimeters", "Inches"])
        
        # Unit conversion if needed
        if height_unit == "Inches":
            plotted_data = example_data['continuous'] / 2.54
            x_label = "Height (inches)"
        else:
            plotted_data = example_data['continuous']
            x_label = "Height (cm)"
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram with KDE
        sns.histplot(plotted_data, bins=bin_count, kde=True, color='skyblue', ax=ax)
        
        # Add mean and median lines
        mean_val = np.mean(plotted_data)
        median_val = np.median(plotted_data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Heights (Continuous Variable)')
        ax.legend()
        
        st.pyplot(fig)
        
        # Interactive experiment
        st.markdown("<div class='example-box'>", unsafe_allow_html=True)
        st.markdown("<h4>Try it yourself!</h4>", unsafe_allow_html=True)
        st.markdown("""
        Generate your own continuous data distribution. Adjust the parameters and observe how 
        the shape of the distribution changes.
        """)
        
        dist_type = st.selectbox("Distribution type:", ["Normal", "Uniform", "Exponential", "Bimodal"])
        sample_size = st.slider("Sample size:", 100, 2000, 500)
        
        if dist_type == "Normal":
            mean = st.slider("Mean:", 0, 100, 50)
            std = st.slider("Standard deviation:", 1, 30, 15)
            custom_data = np.random.normal(mean, std, sample_size)
            dist_desc = f"Normal distribution with mean={mean} and standard deviation={std}"
        elif dist_type == "Uniform":
            low = st.slider("Minimum value:", 0, 50, 10)
            high = st.slider("Maximum value:", 51, 100, 90)
            custom_data = np.random.uniform(low, high, sample_size)
            dist_desc = f"Uniform distribution from {low} to {high}"
        elif dist_type == "Exponential":
            scale = st.slider("Scale parameter (β):", 1, 20, 10)
            custom_data = np.random.exponential(scale, sample_size)
            dist_desc = f"Exponential distribution with β={scale}"
        else:  # Bimodal
            mean1 = st.slider("First peak mean:", 10, 40, 25)
            mean2 = st.slider("Second peak mean:", 60, 90, 75)
            std = st.slider("Standard deviation for both peaks:", 1, 15, 8)
            mix = np.random.choice([0, 1], size=sample_size)
            custom_data = np.where(mix, np.random.normal(mean1, std, sample_size), 
                                 np.random.normal(mean2, std, sample_size))
            dist_desc = f"Bimodal distribution with peaks at {mean1} and {mean2}"
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(custom_data, bins=30, kde=True, color='lightgreen', ax=ax2)
        ax2.set_title(f'Custom {dist_desc}')
        st.pyplot(fig2)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif variable_type == "Categorical":
        st.markdown("<div class='concept-box'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-header'>Categorical Variables</h3>", unsafe_allow_html=True)
        st.markdown("""
        Categorical variables represent groupings or categories without an intrinsic order. They classify
        data into discrete, non-numeric groups with no meaningful numerical relationship between them.
        
        **Examples:** Colors, gender, blood type, country, product category, yes/no responses
        
        **Key characteristics:**
        - Data can be sorted into distinct categories
        - Categories have no inherent order
        - Mathematical operations don't make sense between categories
        - Often represented using text labels (though they can be coded as numbers)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interactive visualization for categorical data
        st.markdown("<h4>Interactive Exploration: Color Preferences</h4>", unsafe_allow_html=True)
        
        # Count frequencies
        cat_counts = pd.Series(example_data['categorical']).value_counts().reset_index()
        cat_counts.columns = ['Color', 'Count']
        
        # Interactive visualization options
        chart_type = st.radio("Select chart type:", ["Bar Chart", "Pie Chart"], horizontal=True)
        
        if chart_type == "Bar Chart":
            sort_order = st.radio("Sort order:", ["Alphabetical", "By Frequency"], horizontal=True)
            if sort_order == "Alphabetical":
                cat_counts = cat_counts.sort_values('Color')
            else:
                cat_counts = cat_counts.sort_values('Count', ascending=False)
                
            # Create chart with actual color representations
            color_map = {
                'Red': '#FF0000',
                'Blue': '#0000FF',
                'Green': '#00CC00',
                'Yellow': '#FFFF00',
                'Purple': '#800080'
            }
            
            chart = alt.Chart(cat_counts).mark_bar().encode(
                x=alt.X('Color:N', sort=None),
                y='Count:Q',
                color=alt.Color('Color:N', scale=alt.Scale(domain=list(color_map.keys()), 
                                                         range=list(color_map.values()))),
                tooltip=['Color', 'Count']
            ).properties(
                title='Distribution of Color Preferences',
                width=600,
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)
            
        else:  # Pie Chart
            fig, ax = plt.subplots(figsize=(10, 10))
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            ax.pie(cat_counts['Count'], labels=cat_counts['Color'], autopct='%1.1f%%', 
                   startangle=90, colors=colors, explode=[0.05] * len(cat_counts))
            ax.set_title('Distribution of Color Preferences')
            st.pyplot(fig)
        
        # Interactive experiment
        st.markdown("<div class='example-box'>", unsafe_allow_html=True)
        st.markdown("<h4>Create your own categorical dataset</h4>", unsafe_allow_html=True)
        
        # Let students create their own categories
        st.markdown("Enter 3-6 categories (one per line):")
        user_categories_input = st.text_area("", "Dogs\nCats\nBirds\nFish", height=150)
        user_categories = [c.strip() for c in user_categories_input.split('\n') if c.strip()]
        
        # Validation
        if len(user_categories) < 3:
            st.warning("Please enter at least 3 categories.")
        else:
            # Create probabilities
            st.markdown("Adjust the probabilities for each category:")
            
            # Initialize with equal probabilities
            default_probs = [round(1/len(user_categories), 2)] * len(user_categories)
            
            # Create sliders for each category
            custom_probs = []
            for i, category in enumerate(user_categories):
                prob = st.slider(f"Probability for {category}:", 0.0, 1.0, default_probs[i], 0.01)
                custom_probs.append(prob)
            
            # Normalize probabilities to sum to 1
            total_prob = sum(custom_probs)
            if total_prob == 0:
                st.error("At least one category must have a probability greater than 0.")
            else:
                norm_probs = [p/total_prob for p in custom_probs]
                
                # Display normalized probabilities
                st.markdown("Normalized probabilities (sum to 1):")
                for cat, prob in zip(user_categories, norm_probs):
                    st.markdown(f"- {cat}: {prob:.2f}")
                
                # Generate data
                sample_size = st.slider("Sample size:", 50, 1000, 300)
                generated_data = np.random.choice(user_categories, size=sample_size, p=norm_probs)
                
                # Display results
                gen_counts = pd.Series(generated_data).value_counts().reset_index()
                gen_counts.columns = ['Category', 'Count']
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Category', y='Count', data=gen_counts, ax=ax3)
                ax3.set_title('Generated Categorical Data')
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
                st.pyplot(fig3)
                
                # Expected vs Actual
                st.markdown("### Expected vs Actual Frequencies")
                comparison_df = pd.DataFrame({
                    'Category': user_categories,
                    'Expected_Proportion': norm_probs,
                    'Expected_Count': [round(p * sample_size) for p in norm_probs]
                })
                
                # Merge with actual counts
                comparison_df = comparison_df.merge(gen_counts, on='Category', how='left')
                comparison_df['Actual_Proportion'] = comparison_df['Count'] / sample_size
                comparison_df['Difference'] = comparison_df['Count'] - comparison_df['Expected_Count']
                
                st.dataframe(comparison_df[['Category', 'Expected_Proportion', 'Actual_Proportion', 
                                          'Expected_Count', 'Count', 'Difference']])
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:  # Ordinal
        st.markdown("<div class='concept-box'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-header'>Ordinal Variables</h3>", unsafe_allow_html=True)
        st.markdown("""
        Ordinal variables are categorical variables with a clear, meaningful order or ranking between
        categories. However, the distances between categories may not be equal or even quantifiable.
        
        **Examples:** Education level, satisfaction ratings, socioeconomic status, pain scales, Likert scales
        
        **Key characteristics:**
        - Categories have a meaningful order/ranking
        - The differences between levels may not be equal or measurable
        - Mathematical operations like addition or subtraction often don't make sense
        - May be represented with numbers, but these are rankings rather than measurements
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interactive visualization for ordinal data
        st.markdown("<h4>Interactive Exploration: Skill Levels</h4>", unsafe_allow_html=True)
        
        # Count frequencies
        ordinal_counts = pd.Series(example_data['ordinal']).value_counts().reset_index()
        ordinal_counts.columns = ['Level', 'Count']
        
        # Create properly ordered index
        correct_order = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
        ordinal_counts['Level_Order'] = ordinal_counts['Level'].apply(lambda x: correct_order.index(x))
        ordinal_counts = ordinal_counts.sort_values('Level_Order')
        
        # Visualization
        color_scheme = st.selectbox(
            "Color scheme:", 
            ["Blues", "Greens", "Purples", "Oranges", "Reds"]
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x='Level', y='Count', data=ordinal_counts, 
                   order=correct_order, palette=color_scheme, ax=ax)
        
        # Add percentage labels
        total = ordinal_counts['Count'].sum()
        for i, bar in enumerate(bars.patches):
            percentage = bar.get_height() / total * 100
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                f'{percentage:.1f}%',
                ha='center',
                fontsize=10
            )
            
        ax.set_title('Distribution of Skill Levels (Ordinal Variable)')
        st.pyplot(fig)
        
        # Interactive experiment: Likert scale
        st.markdown("<div class='example-box'>", unsafe_allow_html=True)
        st.markdown("<h4>Create and analyze a Likert scale survey</h4>", unsafe_allow_html=True)
        
        st.markdown("""
        Likert scales are common ordinal measurements used in surveys. Let's simulate responses to a 
        satisfaction survey using a 5-point Likert scale.
        """)
        
        # Create simulation parameters
        question_count = st.slider("Number of survey questions:", 3, 10, 5)
        respondent_count = st.slider("Number of survey respondents:", 50, 500, 200)
        
        # Generate synthetic survey questions
        topics = ["Product quality", "Customer service", "Ease of use", "Value for money", 
                 "Design", "Features", "Reliability", "Support", "Documentation", "Overall experience"]
        
        selected_topics = topics[:question_count]
        
        # Create a bias parameter (higher = more positive responses)
        response_bias = st.slider("Response bias (higher = more positive responses):", 
                                0.0, 2.0, 1.0, 0.1)
        
        # Generate responses with specified bias
        likert_options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        
        # Adjust probabilities based on bias
        if response_bias < 1.0:
            # Negative bias (more negative responses)
            base_probs = [0.3, 0.3, 0.2, 0.1, 0.1]
        elif response_bias == 1.0:
            # Neutral (balanced responses)
            base_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        else:
            # Positive bias (more positive responses)
            base_probs = [0.1, 0.1, 0.2, 0.3, 0.3]
            
        # Generate synthetic responses
        np.random.seed(42)  # For reproducibility
        survey_data = {}
        
        for i, topic in enumerate(selected_topics):
            # Add some variation between questions
            q_probs = base_probs.copy()
            # Slightly shift probabilities for variety
            shift = np.random.normal(0, 0.05, 5)
            q_probs = [max(0.01, min(0.99, p + s)) for p, s in zip(q_probs, shift)]
            # Normalize probabilities
            q_probs = [p/sum(q_probs) for p in q_probs]
            
            # Generate responses
            survey_data[f"Q{i+1}: {topic}"] = np.random.choice(
                likert_options, 
                size=respondent_count,
                p=q_probs
            )
        
        # Convert to DataFrame
        survey_df = pd.DataFrame(survey_data)
        
        # Show sample of responses
        st.markdown("#### Sample of survey responses")
        st.dataframe(survey_df.head(10))
        
        # Analyze results
        st.markdown("#### Survey Results Analysis")
        
        analysis_type = st.radio(
            "Choose analysis type:",
            ["Response Distribution", "Score Conversion", "Cross-question Analysis"],
            horizontal=True
        )
        
        if analysis_type == "Response Distribution":
            # Select a question
            question = st.selectbox("Select question to analyze:", survey_df.columns)
            
            # Create plot
            likert_order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
            response_counts = survey_df[question].value_counts().reset_index()
            response_counts.columns = ['Response', 'Count']
            
            # Add order index
            response_counts['Order'] = response_counts['Response'].apply(lambda x: likert_order.index(x))
            response_counts = response_counts.sort_values('Order')
            
            # Create stacked bar chart
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            
            # Calculate percentages
            total_responses = response_counts['Count'].sum()
            percentages = [count/total_responses*100 for count in response_counts['Count']]
            
            # Define colors for Likert responses
            likert_colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
            
            # Create horizontal bar
            y_pos = 0
            for i, (response, count, percentage) in enumerate(zip(response_counts['Response'], 
                                                              response_counts['Count'],
                                                              percentages)):
                ax4.barh(y_pos, percentage, color=likert_colors[i], height=0.6)
                # Add text label
                if percentage > 5:  # Only add text if segment is wide enough
                    ax4.text(percentage/2, y_pos, f"{response}: {percentage:.1f}%", 
                           ha='center', va='center', color='black', fontweight='bold')
            
            # Set axis properties
            ax4.set_yticks([])
            ax4.set_xlim(0, 100)
            ax4.set_xlabel('Percentage of Responses')
            ax4.set_title(f'Response Distribution: {question}')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=likert_colors[i], label=likert_order[i]) 
                              for i in range(len(likert_order))]
            ax4.legend(handles=legend_elements, loc='upper center', 
                     bbox_to_anchor=(0.5, -0.15), ncol=5)
            
            st.pyplot(fig4)
            
        elif analysis_type == "Score Conversion":
            st.markdown("""
            We can convert ordinal Likert data to numerical scores for analysis purposes.
            While this conversion should be interpreted carefully, it can provide useful insights.
            """)
            
            # Create mapping
            score_mapping = {
                "Strongly Disagree": 1,
                "Disagree": 2,
                "Neutral": 3,
                "Agree": 4,
                "Strongly Agree": 5
            }
            
            # Convert to numerical scores
            numerical_df = survey_df.replace(score_mapping)
            
            # Calculate statistics
            stats_df = pd.DataFrame({
                'Question': numerical_df.columns,
                'Mean_Score': numerical_df.mean(),
                'Median_Score': numerical_df.median(),
                'Std_Dev': numerical_df.std(),
                'Min': numerical_df.min(),
                'Max': numerical_df.max()
            }).reset_index(drop=True)
            
            # Round to 2 decimal places
            stats_df = stats_df.round(2)
            
            # Display statistics
            st.dataframe(stats_df)
            
            # Visualization of means with confidence intervals
            fig5, ax5 = plt.subplots(figsize=(12, 8))
            
            # Calculate 95% confidence intervals
            ci95_hi = []
            ci95_lo = []
            
            for col in numerical_df.columns:
                mean = numerical_df[col].mean()
                std_err = numerical_df[col].std() / np.sqrt(len(numerical_df))
                ci95_hi.append(mean + 1.96 * std_err)
                ci95_lo.append(mean - 1.96 * std_err)
            
            # Prepare data
            plot_df = pd.DataFrame({
                'Question': [q.split(': ')[1] for q in numerical_df.columns],
                'Mean': numerical_df.mean(),
                'CI_Low': ci95_lo,
                'CI_High': ci95_hi
            })
            
            # Sort by mean rating
            plot_df = plot_df.sort_values('Mean')
            
            # Plot means with error bars
            sns.pointplot(x='Mean', y='Question', data=plot_df, join=False, 
                         errorbar=None, color='darkblue', ax=ax5)
            
            # Add error bars manually
            for i, (idx, row) in enumerate(plot_df.iterrows()):
                ax5.plot([row['CI_Low'], row['CI_High']], [i, i], 'b-', alpha=0.6)
                ax5.plot([row['CI_Low'], row['CI_Low']], [i-0.1, i+0.1], 'b-', alpha=0.6)
                ax5.plot([row['CI_High'], row['CI_High']], [i-0.1, i+0.1], 'b-', alpha=0.6)
            
            # Add reference line at midpoint (3)
            ax5.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
            
            # Set limits to Likert scale range with some padding
            ax5.set_xlim(0.5, 5.5)
            
            # Add grid
            ax5.grid(True, axis='x', alpha=0.3)
            
            # Labels
            ax5.set_title('Mean Scores with 95% Confidence Intervals')
            ax5.set_xlabel('Mean Rating (1-5 scale)')
            ax5.set_ylabel('')
            
            st.pyplot(fig5)
            
        else:  # Cross-question Analysis
            st.markdown("""
            Let's examine correlations between different survey questions. Strong correlations may 
            indicate that questions are measuring related aspects of satisfaction.
            """)
            
            # Convert to numerical for correlation analysis
            score_mapping = {
                "Strongly Disagree": 1,
                "Disagree": 2,
                "Neutral": 3,
                "Agree": 4,
                "Strongly Agree": 5
            }
            numerical_df = survey_df.replace(score_mapping)
            
            # Calculate correlation matrix
            corr_matrix = numerical_df.corr()
            
            # Visualization
            fig6, ax6 = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax6)
            
            # Clean up question labels
            labels = [q.split(': ')[1] for q in corr_matrix.columns]
            ax6.set_xticklabels(labels, rotation=45, ha='right')
            ax6.set_yticklabels(labels, rotation=0)
            
            ax6.set_title('Correlation Between Survey Questions')
            st.pyplot(fig6)
            
            # Interpretation
            st.markdown("""
            #### Interpreting correlations:
            - **Strong positive correlation (>0.7):** Questions likely measure similar concepts
            - **Moderate correlation (0.3-0.7):** Questions have some relationship
            - **Weak correlation (<0.3):** Questions measure mostly different concepts
            - **Negative correlation:** As one score increases, the other tends to decrease
            """)
            
        st.markdown("</div>", unsafe_allow_html=True)

