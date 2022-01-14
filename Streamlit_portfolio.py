
import pandas as pd
from seaborn import palettes
import streamlit as st
import plotly.express as px
from PIL import Image
import plotly.graph_objs as go
import requests
from streamlit_lottie import st_lottie
from urllib.request import urlopen
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from plotly.subplots import make_subplots
import plotly.io as pio
import textwrap
plt.style.use('default')
plt.style.use('seaborn-white')
plt.style.use('seaborn-notebook')
sns.set_palette('Set2')
st.set_option('deprecation.showPyplotGlobalUse', False)

pio.templates.default = "seaborn"

df20 = pd.read_csv('C:/Users/39389/Desktop/DATA VIZ PROJECT/datasets/Merged dataset/2020.csv')
df21 = pd.read_csv('C:/Users/39389/Desktop/DATA VIZ PROJECT/datasets/Merged dataset/2021.csv')
df_ink2 = pd.read_csv('C:/Users/39389/Desktop/DATA VIZ PROJECT/datasets/world-happiness-report.csv')

lcols = [0,1,2,6,7,8,9,10,11]
df20 = df20.iloc[:, lcols]
df21 = df21.iloc[:, lcols]

df20['Year'] = 2020
df21['Year'] = 2021

df = pd.concat([df21, df20])
df.rename(columns={'Country name': 'Country', 
                   'Regional indicator': 'Region', 
                   'Ladder score': 'Happiness score'}, inplace=True)
df['all'] = ''
# display(df)

df20 = df[df.Year == 2020]
df21 = df[df.Year == 2021]
# display(df20)

# create long df

dfM = df.melt(id_vars=['Country', 
                       'Region', 
                       'Year'], 
              value_vars=df.columns[2:-2], 
              var_name='Var', value_name='Value')
dfM['all'] = ''
# display(dfM)

# create wide %change df

dfall = dfM.pivot(index=['Country', 'Region'], columns=['Var', 'Year'], values='Value').reset_index()

scorecols = df.columns[2:-2]
for col in scorecols:
    dfall[(col, '% Ch')] = ((dfall[(col, 2021)] - dfall[(col, 2020)])/dfall[(col, 2020)])*100
# display(dfall.sort_index(axis=1))

dfdiff = dfall[['Country', 'Region']].join(dfall.loc[:, pd.IndexSlice[:, '% Ch']])
# display(dfdiff)

dfdiff.columns = dfdiff.columns.to_flat_index().str.join(' ').str.rstrip(' ')
dfdiff.dropna(inplace=True)
# display(dfdiff)

# create long %change df

dfdiffM = dfdiff.melt(id_vars = dfdiff.columns[0:2], 
                      value_vars = dfdiff.columns[2:], 
                      var_name='Var', value_name='Value'
                     )
# dfdiffM['all'] = ''

# add feature
dfdiffM['% Change'] = dfdiffM.apply(lambda row: 'Positive' if row.Value > 0
                                  else 'Negative', 
                                  axis=1)


# lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# page configuration
st.set_page_config(page_title='Happiness Portfolio', page_icon=':tada:', layout='wide')





lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_ryoaipy4.json")
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title('Happiness portfolio')
        st.subheader("In this portfolio, I will try to represent what happiness means to people, not just in words, but using data from a variety of sources to try to quantify in general what it takes to be happy. ")
    with right_column:
        st_lottie(lottie_coding, height= 300, key= "coding")


data_2021 = pd.read_csv('C:/Users/39389/Downloads/world-happiness-report-2021 (1).csv')
happy_world = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_Z6rQ9X.json")
sad_world = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_sw7ogx3k.json")
with st.container():
    st.header('Survey')
    st.subheader('These are the result from 2021')
    st.text('Click on the columns names to organize the data as you prefer.')
    st.dataframe(data_2021)
    st.write('This is an interactive table, if you want to check the data from which the visualization come from, feel free to do it. The table is always a good form of data viz, for this reason I decided to put it anyway, in order to organize better the CSV file. In case you do not want to waste time reading the table, the results of the first 5 nations and the last 5 can be summarized in the 2 figures below.')
    
    

    left_column, right_column = st.columns(2)
    with left_column:
        st_lottie(happy_world, height= 200, key= "codi")
    with right_column:
        st_lottie(sad_world, height= 200, key= "codin")
    st.write("Just joking, in this portfolio you will understand better which are the most important features that try to explain happiness (backed by data).")
    st.write("[Click here to download the data ](https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021/download)")
    st.write("---")

df_normal = pd.DataFrame(data_2021)

# Bad Viz container
with st.container():
    st.subheader('This Visualization could be improved')
    left_column, right_column = st.columns(2)
    with left_column:
        image = Image.open('C:/Users/39389/Pictures/Bad Vizual.jpg')
        with st.container():
            st.image(image,
                    caption='This is a bad Visualization',
                    use_column_width=True)
            
        with right_column:
            st.write("At first glance, seeing this chart, we notice that the choice of colors is a bit misleading. First, a color palette is used, the lightest color represents the first country for happiness (Finland), while the darkest color represents the last country (South Sudan). The problem with this graphic representation is that it would have been better to use different colors to make the country more recognizable. The best way to recognize the country exactly is to count the bars, fortunately, ordered, so as to understand exactly what shade of the color represents each country. In addition, I would have preferred to put the name of the countries in the X-axis, representing with colors the various features selected. In this way, we would have a clearer comparison between the countries' happiness, directly displaying all metrics belonging to the same country in a stacked bar plot.  ")
    st.write("---")

#Good Viz container
df1 = data_2021.sort_values(by =['Ladder score'],ascending = False).reset_index().head(5)
df2 = data_2021.sort_values(by =['Ladder score'],ascending = True).reset_index().head(5)
result = [df1, df2]
  
df1 = pd.concat(result)
df1 = df1.sort_values(by =['Ladder score'],ascending = False)
tr1 = go.Bar(x = df1['Country name'],
             y = df1['Explained by: Log GDP per capita'],
             name = 'Explained by: Log GDP per capita',
             marker = dict(color = 'SteelBlue'))

tr2 = go.Bar(x = df1['Country name'],
             y = df1['Explained by: Social support'],
             name = 'Explained by: Social support',
             marker = dict(color = 'LightSeaGreen'))

tr3 = go.Bar(x =df1['Country name'],
             y = df1['Explained by: Healthy life expectancy'],
             name = 'Explained by: Healthy life expectancy',
             marker = dict(color = 'DarkSeaGreen'))

tr4 = go.Bar(x = df1['Country name'],
             y = df1['Explained by: Freedom to make life choices'],
             name = 'Explained by: Freedom to make life choices',
             marker = dict(color = 'Gold'))

tr5 = go.Bar(x = df1['Country name'],
             y = df1['Explained by: Generosity'],
             name = 'Explained by: Generosity',
             marker = dict(color = 'FireBrick'))

tr6 = go.Bar(x = df1['Country name'],
             y = df1['Explained by: Perceptions of corruption'],
             name = 'Explained by: Perceptions of corruption',
             marker = dict(color = 'Chocolate'))
             


data_go = [tr1,tr2,tr3,tr4,tr5,tr6]
layout= dict(title = 'The 5 happiest countries vs the 5 most unahappy',
             title_x = 0.5,
             barmode = 'relative',
             title_font = dict(size = 16,color = 'White'),
             xaxis = dict(tickangle = 45)
            
            )

fig_2 = go.Figure(data_go, layout)

with st.container():
    st.subheader('This is an Improved version of the previous visualization')
    st.text('These are the results from 2021. Click on the legend to select or unselect the metrics for a specific comparison!')

    left_column, right_column = st.columns(2)
    with left_column:
        st.write('In order to better visualize each metric belonging to each specific country I decided to use a stacked barplot. I preferred to use data from 2021 so that I would have the most recent data, but both the sense of visualization and the metrics used do not change. The chart is fully interactive, and it is possible to compare the chosen metrics by selecting or deselecting what we prefer to compare, directly from the legend. As we can see, the highest stacked barplot does not belong to the happiest nation. This is due to two reasons. First, because in order to calculate the ladder score that represents the total happiness of a country, is used a formula that gives more weight to certain parameters (e.g. GDP), and less weight to others (e.g. Generosity). Second, not all parameters are positive (e.g. perception of corruption), so the more they are the worst it is. I preferred not to apply the formula used so as not to distort the data, and to analyze objectively what is the right mix for the happiness of a country. ')

    with right_column:
        
        st.plotly_chart(fig_2)
    st.write("---")

country = df_normal['Country name'].unique().tolist()
Ladder_Score = df_normal['Ladder score'].unique().tolist()

st.subheader('Here you can select your tresholds in order to see how many countries are in the happiness range selected.')
Ladder_selection = st.slider('Ladder score:', 
                                min_value= min(Ladder_Score),
                                max_value = max(Ladder_Score),
                                value = (min(Ladder_Score), max(Ladder_Score)))


masks = (df_normal['Ladder score'].between(*Ladder_selection))
number_of_result = df_normal[masks].shape[0] # Drawing Bar chart
mean = np.mean(Ladder_Score)
median = np.median(Ladder_Score)

with st.container():
    st.write('The overall mean is:', mean)
    st.write('The overall median is:', median)
    st.markdown(f'The countries between the ladder score selected are {number_of_result}')
    st.write("---")

with st.container():
    st.subheader("That's a good Visualization")
    left_column, right_column = st.columns(2)
    with left_column:
        image = Image.open('C:/Users/39389/Desktop/DATA VIZ PROJECT/Good visualization.png')
        with st.container():
            st.image(image,
                    
                    use_column_width=True)
    with right_column:
        st.write('This is a great visualization for several reasons. First of all because the author has chosen colors that represent well happiness (green) and unhappiness (red). In addition, the transparency chosen to show the different curves is optimal, in my humble opinion has been chosen a suitable "alpha" and it looks really clean. In a single graph has managed to put 6 plots effectively, optimizing space and making it easy to compare the various graphs. The description is short, straight to the point, and manages to make the viewer understand immediately what it is. The density curves, although simple, are very good to represent this type of data, so the choice is fully apt. In conclusion, I think this chart is very good and easy to read, while at the same time managing, in its simplicity, to make the observer understand a lot about the dataset.') 


        

st.write("[Click here to go to the author of this chart](https://www.kaggle.com/joshuaswords/awesome-eda-2021-happiness-population/notebook)")
st.write("---")

with st.container():
    st.subheader("Visualization of the author's happiness")
    left_column, right_column = st.columns(2)
    with left_column:
        image = Image.open('C:/Users/39389/Downloads/Personal Viz.png')
        with st.container():
            st.image(image,
                    
                    use_column_width=True)
    with right_column:
        
        st.write('This is a nightingale chart, I preferred to use this type of visualization, because it is very easy to read and can effectively show the scale of values also displaying the weight that each value has. For me it is the best choice to be able to show in a single image, what is important to me in order to be happy, and at the same time at what level I am on a scale from 0 to 100 of each metric. Unlike the happiness report, there are no metrics on this scale that can be evaluated negatively (e.g. perception of corruption). For this reason I decided to change the metrics to my liking, in fact I preferred to use instead of the perception of corruption, my vision of society. All values are positive, so the higher they are, the happier I feel from that specific point of view.')
st.write("---")

def regioncompare(var):
    s = df.groupby('Region')[var].median().sort_values(ascending=False)
    with sns.axes_style("whitegrid"):
        plt.figure()
        sns.violinplot(data=df, x=var, y='Region', hue='Year', 
                       order=s.index,
                       split=True, 
                       palette = sns.color_palette()[2:4],
                       inner='quartiles')
        plt.gca().set_yticklabels([textwrap.fill(t.get_text(), 18)  for t in plt.gca().get_yticklabels()])
        plt.gca().legend(loc='upper left')
        plt.gca().set_title(var + ', by region desc')
        plt.gca().set_xlabel("")
        plt.gca().set_ylabel("");
        sns.despine(bottom=True, left=True)

var=scorecols[0]
regioncompare(var)

with st.container():
    st.subheader("How the happiness changed with covid?")
    
    left_column, right_column = st.columns(2)
    with left_column:
        image = Image.open('C:/Users/39389/Desktop/DATA VIZ PROJECT/merged image.png')
        with st.container():
            st.image(image,
                    
                    use_column_width=True)
    with right_column:
        st.write('In order to compare how covid affected happiness, we have to merge 2020 and 2021 datasets. After that, we have to select a good visualization in order to point out the differences, trying to display as many relevant information as possibile. For this reason I decided to use a the violin plot, which shows the full distribution of the data while summarizing the statistics that are contained in a box plot. Horizontal violin plots are very useful if we want to deal with different categories in the same chart, as how the happiness change for every region. You can easily see for example, that in the Western Europe the happiness is almost the same as 2020, while in Sub-Saharian Africa and South Asia, the people are more happy after covid.')

st.write("[Click here to download the data from 2015 to 2021 ](https://www.kaggle.com/mathurinache/world-happiness-report/download)")
st.write("---")


# Maximize data ink ratio
fig_10 = plt.figure(figsize=(6,6),dpi=150)
gs = fig_10.add_gridspec(1, 1)
gs.update(wspace=0.05, hspace=0.27)
ax0 = fig_10.add_subplot(gs[0, 0])


background_color = "#fafafa"
fig_10.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

c1 = 'Jordan'
c2 = 'Bulgaria'

# filter data
df_ink = df_ink2.loc[(df_ink2['Country name']==c1) | (df_ink2['Country name']==c2)]
df_ink = df_ink2.loc[(df_ink2['year']>2008)]
df_ink = pd.pivot_table(df_ink, values='Life Ladder', index=['year'], columns=['Country name'])


# plot the connected scatterplot
plt.plot(df_ink[c1], df_ink[c2], ' ', marker='o',color='black')

# add annotations in every 3 data points with a loop
for line in range(0, df_ink.shape[0], 1):
     plt.annotate(
          df_ink.index[line], 
          (df_ink[c1].iloc[line], df_ink[c2].iloc[line]+0.025) ,
          va='bottom',
          ha='center',fontfamily='monospace'
     )
        

for s in ['top','right']:
    ax0.spines[s].set_visible(False)
    
#ax0.xaxis.set_ticks(np.arange(1, 10, 1))
#ax0.set_ylabel(f"{c2}",fontfamily='monospace',loc= 'top',color='gray')
#ax0.set_xlabel(f"{c1}",fontfamily='monospace',loc='left',color='gray')
ax0.tick_params(axis = 'both', which = 'major', labelsize = 10)
    

ax0.tick_params(axis='both', which='both',left=False, bottom=False,labelbottom=True) 


Xstart, Xend = ax0.get_xlim()
Ystart, Yend = ax0.get_ylim()

ax0.text(Xstart,Yend+0.4,'Comparing happiness between Bulgaria and Jordan over time',fontsize=15,fontweight='bold',color='#323232')
ax0.text(Xstart,Yend+0.15,'Here how the things changed from 2010 to 2020',fontsize=10,fontfamily='monospace',color='gray')

fig_10.text(0.16,0.35,
'''
Over this time period, Bulgaria is the country
 who has improved the most, meanwhile Jordan is
  the one who has deteriorated the most
''',fontfamily='monospace',fontsize=8,color='gray')

# x axis label
plt.xlabel(c1)

# y axis label
plt.ylabel(c2)

# show the graph
max_data_ink = fig_10



with st.container():
    st.subheader("That is a visualization that maximize the data-ink ratio")
    
    left_column, right_column = st.columns(2)
    with left_column:
        st.pyplot(max_data_ink)
       
    with right_column:
        st.write('There are several reasons why you should maximize the data-ink ratio, this is done not only to save your printer toner, but also because a good graphical representation should express the message as simply as possible, without adding fancy stuff to the graph. As you can see, everything in this graph has the function of adding a piece of information, there is nothing that could be removed without losing important data.') 



st.write("[Click here to download the data used for this visualization ](https://www.kaggle.com/gcmadhan/world-happiness-index-report/data?select=world-happiness-report.csv)")
st.write("---")



fig_7 = px.sunburst(data_frame=data_2021,
                  path=["Regional indicator", "Country name"],
                  values="Ladder score",
                  color="Ladder score",
                  width=900, 
                  height=900)




st.subheader('This is a sunburst chart')
st.write('In order to better visualize the entire happiness dataset, I decided to use a Sunburst chart. This type of chart is very useful to visualize every country in the world, divided by region. The radial orientation of this particlar chart is way more efficient compared to an orizontal or vertical visualization. Due to its circular shape, every element in the chart has the same importance, and there is also a small difference in the slice in proportion to the happiness of each nation. The chart is very intuitive to read, and is also interactive, the ladder score of every country can be read just moving the cursor on the chart. Moreover the color palette used, let very easy to see how happy they are in different part of the world without seeing the ladder score, and it is also easy to spot outliers.  ')
st.plotly_chart(fig_7)



   


fig = px.choropleth(data_frame=df_normal,
                    locations="Country name",
                    locationmode="country names",
                    color="Ladder score",
                    title="Happiness score world map")
with st.container():
    st.subheader('World happiness map')

    left_column, right_column = st.columns(2)
    with left_column:
        st.write('This is an interactive world map which shows the happiness. As you can see i have used the same color palette of the sunburst. This kind of visualization is good in order to see all the coutries in a geographical map. Is completely intercative, it is possible to zoom on some specific areas and also, with the use of the cursor, to see the happiness score and the name of the country. The data that are missing are displayed in grey, while the white part is everything that is not a country. ')
      
    with right_column:
        st.plotly_chart(fig)
st.write('---')


# assign data of lists.
data = {'Country': ['Switzerland', 'Germany', 'France', 'Italy', 'Austria', 'Netherlands', 'Norway', 'UK'],
        'Suicide mortality per 100k inhabitants': [10.4,9.5,14.6,5.5,12.4,10.6,11.6,7.3],
        'Health system costs in dollars per inhabitant': [7280,6224,5154,3485,5538,5436,8283,4290],
        'Male life expectancy': [81.7,78.6,79.7,81.2,79.4,80.3,81.1,79.5],
        'Alcohol consumption (in liters per year)': [9.1,10.8,11.6,7.8,12.2,8.3,6.0,9.8],
        'Ladder Score': [7.57, 7.15, 6.69, 6.48,7.26,7.46,7.39,7.06]
        }
# Create DataFrame.
df = pd.DataFrame(data)
# Print the output.

with st.container():
    st.title('Health indicators in a European comparison')
    st.markdown('The goal of this Bar Plot, is to compare different European countries. The data are shown from 0 to 1, where 1 is given to the country that has the best metric about a certain topic, while the other countries have lower values in percentage compared to the first with the value 1. Obviously this visualization is good to understand who has the better metrics compared to other countries but at the same time loses the real data. For this reason, in order not to have a loss of information, it is possible to consult interactively the individual data below. For example Switzerland has 71.23 percent of suicide mortality compared to France. Italy spends 42.07 percent for the health of each citizen compared to what Norway spends. Now you can make your comparisons directly on the chart.') 


    months = df['Country']

    fig_8 = go.Figure()
    fig_8.add_trace(go.Bar(
        x=months,
        y= df['Suicide mortality per 100k inhabitants']/14.6,
        name='Suicide mortality per 100k inhabitants',
        marker_color='#EC114D'
    ))
    fig_8.add_trace(go.Bar(
        x=months,
        y=df['Health system costs in dollars per inhabitant']/8283,
        name='Health system costs in dollars per inhabitant',
        marker_color='#52C231'
    ))
    fig_8.add_trace(go.Bar(
        x=months,
        y=df['Male life expectancy']/81.7,
        name='Male life expectancy',
        marker_color='#1113E4'
    ))
    fig_8.add_trace(go.Bar(
        x=months,
        y=df['Alcohol consumption (in liters per year)']/12.2,
        name='Alcohol consumption (in liters per year)',
        marker_color='#6C2881'
    ))
    fig_8.add_trace(go.Bar(
        x=months,
        y=df['Ladder Score']/7.57,
        name='Ladder Score',
        marker_color='#E1DE31'
    ))

    st.text('Click on the legend to select or unselect the metrics for a specific comparison!')

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig_8.update_layout(barmode='group', xaxis_tickangle=-45)
    st.plotly_chart(fig_8)
    st.write("[Click here to download the data ](https://www.bfs.admin.ch/bfsstatic/dam/assets/15324707/master)")

st.markdown('As we can see the Switzerland is the best for the male life expectancy, and for the happiness score. We can see also that Italy has the lowest suicide rate, while having the lowest happiness score and the lowest health expenditure per capita. ')

st.write('---')

st.subheader('Here you can check the real data in an interactive way')
dropdownII = st.selectbox('Choose your metric', df.columns[df.columns != 'Country'])





fig = px.bar(df, y=dropdownII, x="Country",
            title="Health indicators in a European comparison")

st.plotly_chart(fig)

fig_2 = px.scatter(data_frame=df,
                 x='Ladder Score',
                 y=dropdownII,
                 size='Ladder Score',
                 trendline='ols',
                 color = 'Country'
                 )

if st.button('Click here to show also the scatterplot between the ladder score and the data selected'):
    st.text('Click on the legend to select or unselect countries for a specific comparison!')
    st.plotly_chart(fig_2)

st.markdown('The scatterplot that best explain the correlation between happiness and health is undoubtely the Health system costs in dollars per inhabitant. Infact we can clearly see that the countries that spend more on the health of their citizens are the ones who have the most happy people.')



fuldf = pd.read_csv("C:/Users/39389/Downloads/dff.csv")
 
st.subheader('Documentation on how to improve a Viz')
st.write('In this portfolio we analyzed how some features impact on the happiness of a country (as health expenditure and GDP per capita). Now we want to analyze how the freedome to make life choices impact the happiness of a country. I will show how, improving the same visualization (a scatterplot) we can improve the understanding and the quality of the plot. In order to see the difference you can click 3 different buttons below, that will show how, using the same data, the visualization can change a lot')

fig_1 = px.scatter(data_frame = fuldf,
           x = 'Freedom', 
           y = 'Happiness Score')


if st.button('Click here to show a basic version of the scatterplot'):
    st.plotly_chart(fig_1)

fig_2 = px.scatter(data_frame = fuldf,
           x = 'Freedom', 
           y = 'Happiness Score', 
           color = 'Happiness Score', 
           hover_name = 'Country', 
           title = 'Happiness Scores vs GDP')

st.write('Here there is an improved version of the first plot, where you can see the country name, and we added a color palette to the happiness feature')
if st.button('Click here to show an intermediate version of the scatterplot'):
    st.plotly_chart(fig_2)



fig_3 = px.scatter(data_frame = fuldf,
           x = 'Freedom', 
           y = 'Happiness Score', 
           animation_frame = 'Year',
           animation_group = 'Country',
           size = 'GDP', 
           color = 'Happiness Score', 
           hover_name = 'Country', 
           title = 'Happiness Scores vs GDP')

st.write('Now we make it more informative. We add the size, that represent the GDP per capita, so the bigger is the scatter, the higher is the GDP per capita. Moreover we make also an animation to the chart, in order to dispaly how over the years things changed (from 2015 to 2019). We can easily see a clear correlation among happiness, GDP per capita, and freedom to make life choices.')

if st.button('Click here to show an advanced version of the scatterplot'):
    st.plotly_chart(fig_3)

st.write('---')









st.subheader('Here you will find some useful packages and resourches for data visualization')

st.markdown('**Python Packages**')

st.write("[Matplotlib -> static,interactive, and animated visualization](https://matplotlib.org/)")
st.write("[Seaborn -> build on top of matplotlib and make your viz astonishing ](https://seaborn.pydata.org/)")
st.write("[Plotly -> my favorite, easy to use and useful to make all kind of plots also 3D and interactive. It has some unique graphs that others doesn't have](https://plotly.com/)")
st.write("[Pandas -> all kind of data manipulation, it is a must to clean and make your dataset usable](https://pandas.pydata.org/)")
st.write("[Numpy -> statistics, matrix operations, and much more](https://numpy.org/)")
st.write("[Streamlit -> one of the best alternatives to build and share data apps, very useful and easy to use ](https://streamlit.io/)")


st.write('---')

st.markdown('**Other useful resourches**')
st.write("[Website with all the data viz resourches -> data sources, blogs, tools and packages, chart lists, ispiration websites, etc](https://datavizcatalogue.com/resources.html)")
st.write("[DataIsBeautiful -> best reddit group for data viz and more ](https://www.reddit.com/r/dataisbeautiful/)")
st.write("[Kaggle -> is a social network for data, code, etc. My favourite resource by far](https://www.kaggle.com/)")
st.write("[Opendata.swiss -> Swiss open government data](https://opendata.swiss/de)")
st.write("[Data professor -> YouTube channel with useful videos from beginner to advanced for data visualization](https://www.youtube.com/c/DataProfessor/videos)")
st.write("[Streamlit cheat sheet -> most of the basic Streamlit comands visualized in an efficent way, very useful!](https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/master/app.py)")

st.write('---')

st.subheader('Conclusion and thanks')
st.write('This portfolio has been realized as an university project for the course "**Data Analytics for Finance I & II**" held at **USI** (Università della Svizzera Italiana). My biggest thanks go to Professor **Peter Gruber** for the lessons he gave us, for the critical sense he taught us, and for the guidelines that allowed us to approach this delivery with great freedom and creativity. ')

st.write('Author:')
st.write("[Stefano Viviano](https://www.linkedin.com/in/stefano-viviano-4a20511bb/)")

