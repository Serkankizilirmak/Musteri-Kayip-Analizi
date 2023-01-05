##Import###
import pandas as pd
import seaborn as sns
from chart_studio import plotly
import plotly.offline as pyoff
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt  # For 2D visualization

"""Plotly visualization"""
import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

sns.set_style('whitegrid')

init_notebook_mode(connected=True)  # to display plotly graph offline

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