import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent / 'studentinsight.db'
FEATURES = ['attendance', 'assignment_score', 'midsem_marks']
DISPLAY = {'attendance':'Attendance (%)','assignment_score':'Assignment Score','midsem_marks':'Mid-semester Marks (/20)'}
st.set_page_config(page_title='StudentInsight | KDD Dashboard', page_icon='🎓', layout='wide')

def create_demo_db():
    rng = np.random.default_rng(42)
    ranges = [(75,100,70,100,14,20),(50,85,45,80,8,16),(20,65,20,60,3,12)]
    rows=[]
    for i in range(1,201):
        a,b,c,d,e,f = ranges[(i-1)%3]
        rows.append({'student_id':f'STU{i:03d}','attendance':int(rng.integers(a,b+1)),
                     'assignment_score':int(rng.integers(c,d+1)),'midsem_marks':int(rng.integers(e,f+1))})
    with sqlite3.connect(DB_PATH) as con:
        pd.DataFrame(rows).to_sql('students',con,if_exists='replace',index=False)

def load_db():
    if not DB_PATH.exists(): create_demo_db()
    with sqlite3.connect(DB_PATH) as con:
        tables=pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name='students'",con)
        if tables.empty:
            create_demo_db()
            with sqlite3.connect(DB_PATH) as con2: return pd.read_sql_query('SELECT * FROM students',con2)
        return pd.read_sql_query('SELECT * FROM students',con)

def prepare(raw):
    df=raw.copy()
    aliases={'student_id':['student_id','studentid','id','roll_no','roll_number'],
             'attendance':['attendance','attendance_percent','attendance_percentage'],
             'assignment_score':['assignment_score','assignment','assignment_marks'],
             'midsem_marks':['midsem_marks','mid_sem_marks','midsem','midterm_marks']}
    lower={str(c).strip().lower():c for c in df.columns}; rename={}
    for standard, choices in aliases.items():
        for choice in choices:
            if choice in lower: rename[lower[choice]]=standard; break
    df=df.rename(columns=rename)
    missing=[c for c in FEATURES if c not in df.columns]
    if missing: st.error('Missing required columns: '+', '.join(missing)+'. Required: attendance, assignment_score, midsem_marks.'); st.stop()
    if 'student_id' not in df: df['student_id']=[f'STU{i:03d}' for i in range(1,len(df)+1)]
    for c in FEATURES: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=FEATURES).assign(student_id=lambda x:x.student_id.astype(str)).copy()

st.title('🎓 StudentInsight')
st.subheader('Student Academic Performance Analysis using K-Means Clustering')
st.caption('Data Mining and Knowledge Discovery in Databases (KDD) project')
with st.sidebar:
    st.header('Dashboard Controls')
    source=st.radio('Choose data source',['SQLite demo database','Upload CSV'])
    uploaded=st.file_uploader('Upload student data (.csv)',type=['csv']) if source=='Upload CSV' else None
    k=st.slider('Number of clusters (K)',2,6,3)
    st.caption('Features: attendance, assignment score, and mid-semester marks.')
if source=='Upload CSV':
    if uploaded is None: st.info('Upload a CSV, or select the SQLite demo database.'); st.stop()
    try: raw=pd.read_csv(uploaded)
    except Exception as e: st.error(f'Could not read CSV: {e}'); st.stop()
else: raw=load_db()
df=prepare(raw)
if len(df)<k+1: st.error(f'At least {k+1} valid records are required for K={k}.'); st.stop()
X=df[FEATURES]; scaled=StandardScaler().fit_transform(X)
df['cluster']=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(scaled)
score=silhouette_score(scaled,df.cluster) if df.cluster.nunique()>1 else float('nan')
order=df.groupby('cluster').midsem_marks.mean().sort_values().index.tolist()
labels=['Needs Support','Developing','Performing Well','Strong Performance','Advanced','Top Performance']
label_map={cid:labels[min(i,len(labels)-1)] for i,cid in enumerate(order)}
df['performance_group']=df.cluster.map(label_map)
st.info('The built-in SQLite dataset contains **synthetic demonstration records**, not real student data. Upload a properly formatted CSV to analyse another dataset.')
a,b,c,d=st.columns(4); a.metric('Students analysed',f'{len(df):,}'); b.metric('Clusters',df.cluster.nunique()); c.metric('Average attendance',f"{df.attendance.mean():.1f}%"); d.metric('Silhouette score',f'{score:.3f}' if np.isfinite(score) else 'N/A')
st.markdown('### Explore the clusters')
left,right=st.columns([1,2])
with left:
    group=st.selectbox('Filter by performance group',['All groups']+[label_map[i] for i in order])
    search=st.text_input('Search student ID',placeholder='e.g. STU014')
    st.write('K-Means groups records with similar scaled feature values. Cluster numbers are algorithm labels, not ranks.')
with right:
    x=st.selectbox('X-axis',FEATURES,index=0,format_func=lambda z:DISPLAY[z]); y=st.selectbox('Y-axis',FEATURES,index=2,format_func=lambda z:DISPLAY[z])
    fig=px.scatter(df,x=x,y=y,color='performance_group',hover_data=['student_id',*FEATURES],title='Student clusters',labels={z:DISPLAY[z] for z in FEATURES})
    fig.update_layout(height=430,legend_title_text='Performance group'); st.plotly_chart(fig,use_container_width=True)
filtered=df.copy()
if group!='All groups': filtered=filtered[filtered.performance_group==group]
if search.strip(): filtered=filtered[filtered.student_id.str.contains(search.strip(),case=False,na=False)]
cols=['student_id',*FEATURES,'cluster','performance_group']
st.markdown('### Student records')
st.dataframe(filtered[cols].rename(columns=DISPLAY),use_container_width=True,hide_index=True)
st.download_button('⬇️ Download filtered results (CSV)',filtered[cols].to_csv(index=False).encode('utf-8'),'studentinsight_cluster_results.csv','text/csv')
summary=df.groupby(['cluster','performance_group'],as_index=False).agg(student_count=('student_id','count'),avg_attendance=('attendance','mean'),avg_assignment_score=('assignment_score','mean'),avg_midsem_marks=('midsem_marks','mean')).sort_values('avg_midsem_marks')
st.markdown('### Cluster summary')
st.dataframe(summary.rename(columns={'cluster':'Cluster ID','performance_group':'Interpretation','student_count':'Number of students','avg_attendance':'Average attendance (%)','avg_assignment_score':'Average assignment score','avg_midsem_marks':'Average mid-sem marks (/20)'}).round(2),use_container_width=True,hide_index=True)
bar=px.bar(summary,x='performance_group',y='student_count',color='performance_group',labels={'performance_group':'Performance group','student_count':'Students'},title='Number of students in each cluster'); bar.update_layout(showlegend=False,height=350); st.plotly_chart(bar,use_container_width=True)
with st.expander('KDD workflow and project explanation'):
    st.markdown('''1. **Data selection:** retrieve attendance, assignment scores, and mid-semester marks.\n2. **Preprocessing:** convert features to numeric values and remove records with missing feature values.\n3. **Transformation:** standardise features so their different scales do not dominate distance calculations.\n4. **Data mining:** apply K-Means with the selected K.\n5. **Evaluation:** use silhouette score as one indicator of cluster separation.\n6. **Knowledge presentation:** show summaries, visualisations, and downloadable results.\n\n**Limitation:** these groups are exploratory patterns, not definitive judgments about a student's ability. The default data is synthetic.''')
st.caption('StudentInsight • KDD mini-project • Streamlit, SQLite, pandas, scikit-learn, Plotly')
