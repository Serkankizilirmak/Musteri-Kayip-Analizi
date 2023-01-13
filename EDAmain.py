##Import###
import pandas as pd
import seaborn as sns
from chart_studio import plotly
import plotly.offline as pyoff
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt  # For 2D visualization
import numpy as np
from IPython.display import Markdown, display
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
"""Plotly visualization"""
import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

sns.set_style('whitegrid')

init_notebook_mode(connected=True)  # to display plotly graph offline


###

plt_params = {
    # 'figure.facecolor': 'white',
    'axes.facecolor' : 'white',

    ## to set size
    # 'legend.fontsize': 'x-large',
    # 'figure.figsize': (15, 10),
    # 'axes.labelsize': 'x-large',
    # 'axes.titlesize': 'x-large',
    # 'xtick.labelsize': 'x-large',
    # 'ytick.labelsize': 'x-large'
}

plt.rcParams.update(plt_params)
####
def printmd(string):
    display(Markdown(string))
####
### plot_distribution ### ---1---
df = pd.read_csv("D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\TelcoCustomer(TR).csv")
df["Kayıp Durumu"].replace(to_replace = dict(Var = 1, Yok = 0), inplace = True)
churn = df[(df['Kayıp Durumu'] != 0)]
no_churn = df[(df['Kayıp Durumu'] == 0)]
def plot_distribution(var_select, bin_size):

    tmp1 = churn[var_select]
    tmp2 = no_churn[var_select]
    hist_data = [tmp1, tmp2]
    sns.set(rc={'figure.figsize':(7,7)})
    sns.distplot(hist_data[1], label='Kayıp Durumu : Evet', color='gold', kde=True)
    sns.distplot(hist_data[0], label='Kayıp Durumu : Hayır', color='lightblue', kde=True)
    sns.set(rc={'figure.figsize':(9,7)})
    plt.xlabel(var_select)
    plt.ylabel('Dağılım Yoğunluğu')
    plt.title(var_select + ' Dağılımı')
    plt.legend()
    plt.savefig("D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\Çıktılar\Dağılım1\{}.png".format(var_select + "(Dağılımı1)"))
    plt.show()

### plot_distribution_num ### ---2---
def plot_distribution_num(data_select) :
    sns.set_style("ticks")
    s = sns.FacetGrid(df, hue = 'Kayıp Durumu',aspect = 2.5, palette ={0 : 'lightblue', 1 : 'gold'})
    s.map(sns.kdeplot, data_select, shade = True, alpha = 0.8)
    s.set(xlim=(0, df[data_select].max()))
    s.add_legend()
    s.set_axis_labels(data_select, 'proportion')
    s.fig.suptitle(data_select)
    plt.savefig("D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\Çıktılar\Dağılım2\{}.png".format(data_select + "(Dağılımı2)"))
    plt.show()
### barplot ### ---3---
def barplot(var_select, x_no_numeric) :
    df["Kayıp Durumu"].replace(to_replace = dict(Var = 1, Yok = 0), inplace = True)
    tmp1 = df[(df['Kayıp Durumu'] != 0)]
    tmp2 = df[(df['Kayıp Durumu'] == 0)]
    tmp3 = pd.DataFrame(pd.crosstab(df[var_select],df['Kayıp Durumu']), )
    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100
    if x_no_numeric == True  :
        tmp3 = tmp3.sort_values(1, ascending = False)

    trace1 = go.Bar(
        x=tmp1[var_select].value_counts().keys().tolist(),
        y=tmp1[var_select].value_counts().values.tolist(),
        text=tmp1[var_select].value_counts().values.tolist(),
        textposition = 'auto',
        name='Kayıp Durumu : Var',opacity = 0.8, marker=dict(
        color='gold',
        line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp2[var_select].value_counts().keys().tolist(),
        y=tmp2[var_select].value_counts().values.tolist(),
        text=tmp2[var_select].value_counts().values.tolist(),
        textposition = 'auto',
        name='Kayıp Durumu : Yok', opacity = 0.8, marker=dict(
        color='lightblue',
        line=dict(color='#000000',width=1)))

    trace3 =  go.Scatter(
        x=tmp3.index,
        y=tmp3['Attr%'],
        yaxis = 'y2',
        name='% Kayıp Durumu', opacity = 0.6, marker=dict(
        color='black',
        line=dict(color='#000000',width=0.5
        )))

    layout = dict(title =  str(var_select),  autosize = False,
                        height  = 500,
                        width   = 800,
              xaxis=dict(),
              yaxis=dict(title= 'Sayı'),
              yaxis2=dict(range= [-0, 75],
                          overlaying= 'y',
                          anchor= 'x',
                          side= 'right',
                          zeroline=False,
                          showgrid= False,
                          title= '% Kayıp Durumu'
                         ))
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    pio.write_html(fig, file='D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\Çıktılar\Dağılım3\{}.html'.format(var_select + "(Dağılımı3)"))
    iplot(fig)

### plot_distribution_cat ### ---4---
def plot_distribution_cat(feature1,feature2, df):
    plt.figure(figsize=(18,5))
    plt.subplot(121)
    s = sns.countplot(x = feature1, hue='Kayıp Durumu', data = df,
                      palette = {"Yok" : 'lightblue', "Var" :'gold'}, alpha = 0.8,
                      linewidth = 0.4, edgecolor='grey')
    s.set_title(feature1)
    for p in s.patches:
        s.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+30))

    plt.subplot(122)
    s = sns.countplot(x = feature2, hue='Kayıp Durumu', data = df,
                      palette = {"Yok" : 'lightblue', "Var" :'gold'}, alpha = 0.8,
                      linewidth = 0.4, edgecolor='grey')
    s.set_title(feature2)
    for p in s.patches:
        s.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+30))
    plt.savefig("D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\Çıktılar\Dağılım4\{} ve {} (Dağılımı4).png".format(feature1, feature2))
    plt.show()
### binning_feature###

def binning_feature(feature):
    plt.hist(df[feature])

    # set x/y labels and plot title
    plt.xlabel(f"{feature.title()}")
    plt.ylabel("Count")
    plt.title(f"{feature.title()} Bins")
    plt.show()

    bins = np.linspace(min(df[feature]), max(df[feature]), 4)

    printmd("**Value Range**")

    printmd(f"Low ({bins[0] : .2f} - {bins[1]: .2f})")
    printmd(f"Medium ({bins[1]: .2f} - {bins[2]: .2f})")
    printmd(f"High ({bins[2]: .2f} - {bins[3]: .2f})")

    group_names = ['Low', 'Medium', 'High']

    df.insert(df.shape[1]-1,f'{feature}-binned', pd.cut(df[feature], bins, labels=group_names, include_lowest=True))
    display(df[[feature, f'{feature}-binned']].head(10))


    # count values
    printmd("<br>**Binning Distribution**<br>")
    display(df[f'{feature}-binned'].value_counts())


    # plot the distribution of each bin
    plt.bar(group_names, df[f'{feature}-binned'].value_counts())
    # px.bar(data_canada, x='year', y='pop')

    # set x/y labels and plot title
    plt.xlabel(f"{feature.title()}")
    plt.ylabel("Count")
    plt.title(f"{feature.title()} Bins")
    plt.show()

#### make_subplots2 ####
def make_subplots2(feature1,feature2,feature3,feature4):
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]])

    fig.add_trace(go.Pie(labels=df[feature1].value_counts().index,
                         values=df[feature1].value_counts().values, name=feature1),
                  1, 1)
    fig.add_trace(go.Pie(labels=df[feature2].value_counts().index,
                         values=df[feature2].value_counts().values, name=feature2),
                  1, 2)
    fig.add_trace(go.Pie(labels=df[feature3].value_counts().index,
                         values=df[feature3].value_counts().values, name=feature3),
                  2, 1)
    fig.add_trace(go.Pie(labels=df[feature4].value_counts().index,
                         values=df[feature4].value_counts().values, name=feature4),
                  2, 2)

    # donut-like pie chart
    fig.update_traces(hole=.5, hoverinfo="label+percent")

    fig.update_layout(

        # Add annotations in the center of the donut pies.
        annotations=[dict(text=feature1, x=0.195, y=0.85, font_size=20, showarrow=False),
                     dict(text=feature2, x=0.804, y=0.86, font_size=20, showarrow=False),
                     dict(text=feature3, x=0.192, y=0.18, font_size=20, showarrow=False),
                     dict(text=feature4, x=0.805, y=0.18, font_size=20, showarrow=False)])
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.show()

## pie_churn ####

def pie_churn(feature):
    trace = go.Pie(labels = ['Kayıp Durumu : Hayır', 'Kayıp Durumu : Evet'], values = df[feature].value_counts(),
                   textfont=dict(size=15), opacity = 0.8,
                   marker=dict(colors=['lightblue','gold'],
                               line=dict(color='#000000', width=1.5)))


    layout = dict(title =  'Kayıp Durumu Değişkeninin Dağılımı',
                            autosize = False,
                            height  = 500,
                            width   = 800)

    fig = dict(data = [trace], layout=layout)
    iplot(fig)
## pie ##
def pie(feature):
    display(px.pie(df[feature].value_counts().reset_index().rename(columns={'index':'Type'}), values=feature, names='Type', title=feature + "Distribution" ))

## box_plot ##
def box_plot(feature):
    sns.boxplot(x=df[feature])
    plt.show()

### density ###
def density(feature):
    """Plots histogram and density plot of a variable."""

    # Create subplot object
    fig = make_subplots(
        rows=2,
        cols=1,
        print_grid=False,
        subplot_titles=(
        f"Distribution of {feature.name} with Histogram", f"Distribution of {feature.name} with Density Plot"))

    # This is a count histogram
    fig.add_trace(
        go.Histogram(
            x=feature,
            hoverinfo="x+y"
        ),
        row=1, col=1)

    # This is a density histogram
    fig.add_trace(
        go.Histogram(
            x=feature,
            hoverinfo="x+y",
            histnorm="density"
        ),
        row=2, col=1)

    # Update layout
    fig.layout.update(
        height=800,
        width=870,
        hovermode="closest"
    )

    # Update axes
    fig.layout.yaxis1.update(title="<b>Abs Frequency</b>")
    fig.layout.yaxis2.update(title="<b>Density(%)</b>")
    fig.layout.xaxis2.update(title=f"<b>{feature.name}</b>")
    return fig.show()


### plot_counting_distribution ###

### Data Sets ####
df = pd.read_csv("D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\TelcoCustomer(TR)_binned.csv")


def plot_counting_distribution(cardinality_value):
    # label encoding binary columns
    le = LabelEncoder()

    tmp_churn = df[df['Kayıp Durumu'] == 'Var']
    tmp_no_churn = df[df['Kayıp Durumu'] == 'Yok']

    selected_columns = df.nunique()[df.nunique() == cardinality_value].keys()

    for col in selected_columns:
        tmp_churn[col] = le.fit_transform(tmp_churn[col])

    data_frame_x = tmp_churn[selected_columns].sum().reset_index()
    data_frame_x.columns = ["feature", "Var"]
    data_frame_x["Yok"] = tmp_churn.shape[0] - data_frame_x["Var"]
    data_frame_x = data_frame_x[data_frame_x["feature"] != "Kayıp Durumu"]

    # count of 1's(yes)
    trace1 = go.Scatterpolar(r=data_frame_x["Var"].values.tolist(),
                             theta=data_frame_x["feature"].tolist(),
                             fill="toself", name="Churn 1's",
                             mode="markers+lines", visible=True,
                             marker=dict(size=5)
                             )

    # count of 0's(No)
    trace2 = go.Scatterpolar(r=data_frame_x["Yok"].values.tolist(),
                             theta=data_frame_x["feature"].tolist(),
                             fill="toself", name="Churn 0's",
                             mode="markers+lines", visible=True,
                             marker=dict(size=5)
                             )
    for col in selected_columns:
        tmp_no_churn[col] = le.fit_transform(tmp_no_churn[col])

    data_frame_x = tmp_no_churn[selected_columns].sum().reset_index()
    data_frame_x.columns = ["feature", "Var"]
    data_frame_x["Yok"] = tmp_no_churn.shape[0] - data_frame_x["Var"]
    data_frame_x = data_frame_x[data_frame_x["feature"] != "Kayıp Durumu"]

    # count of 1's(yes)
    trace3 = go.Scatterpolar(r=data_frame_x["Var"].values.tolist(),
                             theta=data_frame_x["feature"].tolist(),
                             fill="toself", name="NoChurn 1's",
                             mode="markers+lines", visible=False,
                             marker=dict(size=5)
                             )

    # count of 0's(No)
    trace4 = go.Scatterpolar(r=data_frame_x["Yok"].values.tolist(),
                             theta=data_frame_x["feature"].tolist(),
                             fill="toself", name="NoChurn 0's",
                             mode="markers+lines", visible=False,
                             marker=dict(size=5)
                             )

    data = [trace1, trace2, trace3, trace4]

    updatemenus = list([
        dict(active=0,
             x=-0.15,
             buttons=list([
                 dict(
                     label='Churn Dist',
                     method='update',
                     args=[{'visible': [True, True, False, False]},
                           {'title': f'Customer Churn Binary Counting Distribution'}]),

                 dict(
                     label='No-Churn Dist',
                     method='update',
                     args=[{'visible': [False, False, True, True]},
                           {'title': f'No Customer Churn Binary Counting Distribution'}]),

             ]),
             )
    ])

    layout = dict(title='ScatterPolar Distribution of Churn and Non-Churn Customers (Select from Dropdown)',
                  showlegend=False,
                  updatemenus=updatemenus)

    fig = dict(data=data, layout=layout)

    pio.show(fig)
