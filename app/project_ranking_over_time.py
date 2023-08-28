import pandas as pd
import streamlit as st
import pickle
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

with open('ep_data/yearly_score_dict.pkl', 'rb') as pickle_file:
    yearly_score_dict = pickle.load(pickle_file)
    
combined_df = pd.DataFrame()

for year, df in yearly_score_dict.items():
    df['year'] = year
    df['year_in_dt'] = pd.to_datetime(year, format='%Y')
    combined_df = pd.concat([combined_df, df], ignore_index=True)
    
scaler = MinMaxScaler()

def calculate_scores(df):
    df[['page_rank', 'betweenness_centrality', 'closeness_centrality']] = scaler.fit_transform(df[['page_rank', 'betweenness_centrality', 'closeness_centrality']])
    df['total_score'] = df.apply(lambda row: row.page_rank + row.betweenness_centrality + row.closeness_centrality, axis = 1)
    return df

scores = calculate_scores(combined_df)
scores = scores.sort_values(by='year')

unique_repos = scores['repo'].unique()
color_map = {repo: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
             for i, repo in enumerate(unique_repos)}

fig = px.line(scores, x='year', y='page_rank', color='repo', markers=True,
              title='Rank Changes Over Years', color_discrete_map=color_map)

fig.show()

# Streamlit app
st.title('Rank Changes Over Years based on Page Rank')
st.plotly_chart(fig)

fig = px.line(scores, x='year', y='betweenness_centrality', color='repo', markers=True,
              title='Rank Changes Over Years', color_discrete_map=color_map)

# Show the plot
fig.show()

# Streamlit app
st.title('Rank Changes Over Years based on Betweenness Centrality')
st.plotly_chart(fig)

fig = px.line(scores, x='year', y='closeness_centrality', color='repo', markers=True,
              title='Rank Changes Over Years', color_discrete_map=color_map)

# Show the plot
fig.show()

# Streamlit app
st.title('Rank Changes Over Years based on Closeness Centrality')
st.plotly_chart(fig)

fig = px.line(scores, x='year', y='total_score', color='repo', markers=True,
              title='Total Score Changes Over Years', color_discrete_map=color_map)

# Show the plot
fig.show()

# Streamlit app
st.title('Rank Changes Over Years based on average of all scores')
st.plotly_chart(fig)

fig = px.bar(scores, x='year', y='total_score', color='repo', title='Project Ranking Over Time')
fig.show()

# Streamlit app
st.title('Top ranked projects over time')
st.plotly_chart(fig)

totals_df = scores.groupby('repo')['total_score'].sum().reset_index()
top = totals_df.sort_values(by='total_score', ascending=False)
top_20 = top.head(20)
bottom = totals_df.sort_values(by='total_score', ascending=True)
bottom_20 = bottom.head(20)

fig = px.bar(top_20, x='repo', y='total_score', title='Project Ranking Over Time')
fig.show()

# Streamlit app
st.title('Projects central to the ecosystem')
st.plotly_chart(fig)

fig = px.bar(bottom_20, x='repo', y='total_score', title='Project Ranking Over Time')
fig.show()

# Streamlit app
st.title('Projects in the exterior of the ecosystem')
st.plotly_chart(fig)


