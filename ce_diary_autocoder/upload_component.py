import base64
import datetime
import io
import os
import urllib

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support as score

from ce_diary_autocoder.controller import parse_contents, graph_update

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([

    # Hidden div inside the app that stores the intermediate value,
    html.Div(id='intermediate-value', style={'display': 'none'}),
    # Heading
    html.H3(children='Diary Autocoder'),

    # Upload file button
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '90%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    # status of upload (success or failure)
    html.P(id='upload-status'),

    # Table showing processed file
    html.Div(id='output-data-upload'),

    # Performance Metrics
    html.Div(
        html.H5(children='Quick Stats')
        ),

    html.Div(id='basic-stats'),

    html.Hr(),  # horizontal line

    # Button to download processed file in xlsx format
    html.Div([
    html.A(
        html.Button(
            'Download Excel',
            id='download-button',
            ),
            id='download-link',
            download="new_file.xlsx",
            href="",
            target="_blank",
            style={'padding': 10}
            )]),

    html.Div([
        html.H4(children='Item Errors'),
        # error graph
        dcc.Graph(id='my-graph', animate=True)
    ], style={'width':'49%',
              'display':'inline-block',
              'margin-left':'auto',
              'margin-right':'auto',
              'vertical-align':'left',
              'padding': '0 20'
              }),

    html.Div([
        html.H4(children='Item Drill Down'),
        html.H5(children='Double click in graph to resize.'),
        # error graph drill down
        dcc.Graph(id='x-series',animate=True)
    ], style={'width':'49%',
              'display':'inline-block',
              'margin-left':'auto',
              'margin-right':'auto'}),


    html.Hr(), #horizontal line

    html.Div(children='''Created By Michell Li, 2019.''')

])

# Hidden Div Callback to store reusable dataframe
@app.callback([Output('intermediate-value', 'children'),
               Output('upload-status','children')],
                    [Input('upload-data', 'contents')],
                    [State('upload-data', 'filename')])
def clean_data(list_of_contents,list_of_names):
    if list_of_contents is not None:
        try:
            global_df = [parse_contents(c, n) for c, n in
                        zip(list_of_contents, list_of_names)]
            global_df = global_df[0]
            return global_df.to_json(date_format='iso', orient='split'), 'Upload Success'
        except Exception as e:
            print(e)
            return None, 'Upload failed, please try again'

    else:
        raise PreventUpdate

# Generate table from processed dataframe
@app.callback(Output('output-data-upload', 'children'),
              [Input('intermediate-value', 'children')])
def update_output(jsonified_cleaned_data):

    if jsonified_cleaned_data is not None:
        global_df = pd.read_json(jsonified_cleaned_data, orient='split')

        # drop unclassifiable columns
        if 'Probability 3' in global_df.columns:
            global_df = global_df.drop(global_df.columns[-6:],axis=1)

        children= [html.Div([

            dash_table.DataTable(
                data=global_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in global_df.columns],
                filter_action='native',
                sort_action='native',
                sort_mode='multi',
                page_size=10,
                style_table={
                'maxHeight':'300',
                'overflowY':'scroll'
                }
            )
        ])]

        return children

# Evaluation Metrics
@app.callback(Output('basic-stats','children'),
             [Input('intermediate-value','children')])
def calculate_metrics(jsonified_cleaned_data):
    if jsonified_cleaned_data is not None:
        data = pd.read_json(jsonified_cleaned_data, orient='split')

        # Precision, recall, fscore, support
        precision,recall,fscore,support = score(data['ITEM'].astype('str'),data['Y_PRED'].astype('str'),
                                                average ='weighted')

        # Accuracy
        accuracy = round((data['Y_PRED'] == data['ITEM']).sum() / data.shape[0], 2)

        # % manually coded
        manual_df = data[(data['AUTO_MAN']==2)]

        perc_manual = round(manual_df.shape[0] / data.shape[0] * 100 , 2)

        # % manually coded error
        perc_manual_error = round((manual_df['ITEM'] != manual_df['Y_PRED']).sum() / manual_df.shape[0] * 100, 2)

        # % unclassifiables
        perc_unclassifiable = round((data['ITEM'].astype('int')>899999).sum() / data.shape[0] * 100, 2)

        return html.Div([
                    html.P('Accuracy: {}'.format(accuracy)),
                    html.P('Precision: {}'.format(round(precision,2))),
                    html.P('Recall: {}'.format(round(recall,2))),
                    html.P('F-Score: {}'.format(round(fscore,2))),
                    html.P('Manually Coded: {}%'.format(perc_manual)),
                    html.P('Error in Manual Coding: {}%'.format(perc_manual_error)),
                    html.P('Unclassifiable: {}%'.format(perc_unclassifiable))
                    ])

# Download link to xlsx file
@app.callback(Output('download-link', 'href'),
    [Input('intermediate-value', 'children')])
def update_download_link(jsonified_cleaned_data):
    if jsonified_cleaned_data is not None:
        #load json dataset
        full_dataset = pd.read_json(jsonified_cleaned_data, orient='split')

        # Filter out unclassifiables
        unclassifiables_df = full_dataset[(full_dataset['ITEM'].astype('int') > 899999)]
        unclassifiables_df = unclassifiables_df.drop(columns=['Y_PRED'])
        # drop the unclassifiable columns for datasets other than EMLS
        if 'Probability 3' in full_dataset.columns:
            data = full_dataset.drop(full_dataset.columns[-6:],axis=1)
        else:
            data = full_dataset

        #errors DataFrame
        error_mask = (data['ITEM']!=data['Y_PRED'])
        error_df = data[error_mask]

        #manually coded DataFrame
        manual_mask = (data['AUTO_MAN'] == 2)
        manual_df = data[manual_mask]
        manual_error_yes = manual_df[(manual_df['ITEM'] != manual_df['Y_PRED'])]
        manual_error_no = manual_df[(manual_df['ITEM'] == manual_df['Y_PRED'])]
        manual_error_yes['MATCH'] = 'YES'
        manual_error_no['MATCH'] = 'NO'
        manual_df = pd.concat([manual_error_yes,manual_error_no],axis=0)

        #write to xlsx
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        error_df.to_excel(writer, sheet_name='Errors')
        manual_df.to_excel(writer,sheet_name='Manually Coded')
        unclassifiables_df.to_excel(writer,sheet_name='Unclassifiables')
        data.to_excel(writer,sheet_name='Full Dataset')
        writer.save()
        xlsx_io.seek(0)

        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        data = base64.b64encode(xlsx_io.read()).decode("utf-8")
        href_data_downloadable = f'data:{media_type};base64,{data}'
        return href_data_downloadable

@app.callback(Output('my-graph', 'figure'),
    [Input('intermediate-value','children')])
def update_graph(jsonified_cleaned_data):

    if jsonified_cleaned_data is not None:
        data = pd.read_json(jsonified_cleaned_data, orient='split')

        #get data for x axis and y axis
        x_val, y_val = graph_update(data)

        return {
            'data': [go.Bar(x=[1,2,3,4,5],
                                y=y_val[-5:]),
                ],
            'layout':go.Layout(
                            xaxis=go.layout.XAxis(
                                  tickmode='array',
                                  tickvals=[1,2,3,4,5],
                                  ticktext=x_val[-5:]),
                            yaxis=go.layout.YAxis(range=[0,np.max(y_val)]),

                            hovermode='closest')   }
    return {'data':[go.Bar(x=[1,2,3,4,5],y=[0,0,0,0,0]),

                ]}

@app.callback(
    dash.dependencies.Output('x-series', 'figure'),
    [dash.dependencies.Input('my-graph', 'hoverData'),
     dash.dependencies.Input('intermediate-value','children')])
def update_x_timeseries(hoverData, jsonified_cleaned_data):
    print(hoverData)

    #read in dataset
    if jsonified_cleaned_data is not None and hoverData is not None:
        data = pd.read_json(jsonified_cleaned_data, orient='split')

        # get data for x axis and y axis
        x_val, y_val = graph_update(data)

        # get item code user is hovering over
        idx_hover_item = hoverData['points'][0]['x']
        hover_item_code = x_val[idx_hover_item-6]

        print("hover item code", hover_item_code)
        # create new dataframe with just item codes the user wants to see
        error_df = data[(data['ITEM'] != data['Y_PRED'])]
        hover_error_df = error_df[error_df['ITEM'] == hover_item_code]

        print("hover error df",hover_error_df['ITEM'])

        # set x axis and y axis variables
        x_val_hover = list(hover_error_df['Y_PRED'].value_counts().index)[::-1]
        y_val_hover = list(hover_error_df['Y_PRED'].value_counts())[::-1]
        x_ranges = list(range(len(x_val_hover)))
        print(x_val_hover)
        print(hover_error_df['Y_PRED'].value_counts())

        if len(x_ranges)>5:
            x_ranges = x_ranges[-5:]
            y_val_hover = y_val_hover[-5:]
            x_val_hover = x_val_hover[-5:]

        print(hoverData)
        print(x_ranges)
        print(y_val_hover)
        print(x_val_hover)
        return {
            'data': [go.Bar(x=x_ranges,
                                y=y_val_hover),
                ],
            'layout':go.Layout(
                            xaxis=go.layout.XAxis(
                                  tickmode='array',
                                  tickvals=x_ranges,
                                  ticktext=x_val_hover),
                            yaxis=go.layout.YAxis(range=[0,np.max(y_val_hover)]),
                            hovermode='closest')   }
    return {'data':[go.Bar(x=[1,2,3,4,5],y=[0,0,0,0,0]),

                ]}

def main():
    print("Dashboard loading...")
    print()
    app.run_server(debug=True, use_reloader=False)
if __name__ == '__main__':
    main()
