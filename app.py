#import io
import pickle
from flask import Flask, jsonify, request
from sklearn.metrics import mean_absolute_error
import pandas as pd
# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

feature_col_fill = pickle.load(open('feat_col_fill.pkl','rb'))
best_xgb = pickle.load(open('xgb_model.pkl','rb'))

def final_fun_1():
    ''' Final function 1 to get the input data and predict the y values'''

    #x_column = "Id,Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Feature_7,Feature_8,Feature_9,Feature_10,Feature_11,Feature_12,Feature_13,Feature_14,Feature_15,Feature_16,Feature_17,Feature_18,Feature_19,Feature_20,Feature_21,Feature_22,Feature_23,Feature_24,Feature_25,Ret_MinusTwo,Ret_MinusOne,Ret_2,Ret_3,Ret_4,Ret_5,Ret_6,Ret_7,Ret_8,Ret_9,Ret_10,Ret_11,Ret_12,Ret_13,Ret_14,Ret_15,Ret_16,Ret_17,Ret_18,Ret_19,Ret_20,Ret_21,Ret_22,Ret_23,Ret_24,Ret_25,Ret_26,Ret_27,Ret_28,Ret_29,Ret_30,Ret_31,Ret_32,Ret_33,Ret_34,Ret_35,Ret_36,Ret_37,Ret_38,Ret_39,Ret_40,Ret_41,Ret_42,Ret_43,Ret_44,Ret_45,Ret_46,Ret_47,Ret_48,Ret_49,Ret_50,Ret_51,Ret_52,Ret_53,Ret_54,Ret_55,Ret_56,Ret_57,Ret_58,Ret_59,Ret_60,Ret_61,Ret_62,Ret_63,Ret_64,Ret_65,Ret_66,Ret_67,Ret_68,Ret_69,Ret_70,Ret_71,Ret_72,Ret_73,Ret_74,Ret_75,Ret_76,Ret_77,Ret_78,Ret_79,Ret_80,Ret_81,Ret_82,Ret_83,Ret_84,Ret_85,Ret_86,Ret_87,Ret_88,Ret_89,Ret_90,Ret_91,Ret_92,Ret_93,Ret_94,Ret_95,Ret_96,Ret_97,Ret_98,Ret_99,Ret_100,Ret_101,Ret_102,Ret_103,Ret_104,Ret_105,Ret_106,Ret_107,Ret_108,Ret_109,Ret_110,Ret_111,Ret_112,Ret_113,Ret_114,Ret_115,Ret_116,Ret_117,Ret_118,Ret_119,Ret_120"
    #x_df = pd.read_csv(io.StringIO(X), names=x_column)
    
    X = request.form.to_dict()
    x_df = pd.read_csv(X['input'])

    
    ''' Populating ordinal columns(Probably rank columns) into a list '''
    ordinal_columns = []
    for i in range(1,26):
        col = 'Feature_'+str(i)
        distinct_count = len(x_df.eval(col).value_counts())
        if distinct_count < 11:
            ordinal_columns.append(col)
    
    for ord_col in ordinal_columns:
        x_df[ord_col] = x_df[ord_col].fillna(0)

    
    for feat_col in x_df:
        if 'Feature_' or 'Ret_' in feat_col:
            x_df[feat_col] = x_df[feat_col].fillna(feature_col_fill)

    x_df = x_df.drop(columns=['Id','Feature_1', 'Feature_2','Feature_4','Feature_10','Feature_20'])
    y = best_xgb.predict(x_df)
    print(y[0][3])

    return jsonify({'return_value': str(y[0][3])})




@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/final_fun_2', methods=['POST'])
def final_fun_2():
    ''' Final function 2 to input X and y values and to compute target metric for a row '''
   # column = "Id,Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Feature_7,Feature_8,Feature_9,Feature_10,Feature_11,Feature_12,Feature_13,Feature_14,Feature_15,Feature_16,Feature_17,Feature_18,Feature_19,Feature_20,Feature_21,Feature_22,Feature_23,Feature_24,Feature_25,Ret_MinusTwo,Ret_MinusOne,Ret_2,Ret_3,Ret_4,Ret_5,Ret_6,Ret_7,Ret_8,Ret_9,Ret_10,Ret_11,Ret_12,Ret_13,Ret_14,Ret_15,Ret_16,Ret_17,Ret_18,Ret_19,Ret_20,Ret_21,Ret_22,Ret_23,Ret_24,Ret_25,Ret_26,Ret_27,Ret_28,Ret_29,Ret_30,Ret_31,Ret_32,Ret_33,Ret_34,Ret_35,Ret_36,Ret_37,Ret_38,Ret_39,Ret_40,Ret_41,Ret_42,Ret_43,Ret_44,Ret_45,Ret_46,Ret_47,Ret_48,Ret_49,Ret_50,Ret_51,Ret_52,Ret_53,Ret_54,Ret_55,Ret_56,Ret_57,Ret_58,Ret_59,Ret_60,Ret_61,Ret_62,Ret_63,Ret_64,Ret_65,Ret_66,Ret_67,Ret_68,Ret_69,Ret_70,Ret_71,Ret_72,Ret_73,Ret_74,Ret_75,Ret_76,Ret_77,Ret_78,Ret_79,Ret_80,Ret_81,Ret_82,Ret_83,Ret_84,Ret_85,Ret_86,Ret_87,Ret_88,Ret_89,Ret_90,Ret_91,Ret_92,Ret_93,Ret_94,Ret_95,Ret_96,Ret_97,Ret_98,Ret_99,Ret_100,Ret_101,Ret_102,Ret_103,Ret_104,Ret_105,Ret_106,Ret_107,Ret_108,Ret_109,Ret_110,Ret_111,Ret_112,Ret_113,Ret_114,Ret_115,Ret_116,Ret_117,Ret_118,Ret_119,Ret_120,Ret_121,Ret_122,Ret_123,Ret_124,Ret_125,Ret_126,Ret_127,Ret_128,Ret_129,Ret_130,Ret_131,Ret_132,Ret_133,Ret_134,Ret_135,Ret_136,Ret_137,Ret_138,Ret_139,Ret_140,Ret_141,Ret_142,Ret_143,Ret_144,Ret_145,Ret_146,Ret_147,Ret_148,Ret_149,Ret_150,Ret_151,Ret_152,Ret_153,Ret_154,Ret_155,Ret_156,Ret_157,Ret_158,Ret_159,Ret_160,Ret_161,Ret_162,Ret_163,Ret_164,Ret_165,Ret_166,Ret_167,Ret_168,Ret_169,Ret_170,Ret_171,Ret_172,Ret_173,Ret_174,Ret_175,Ret_176,Ret_177,Ret_178,Ret_179,Ret_180,Ret_PlusOne,Ret_PlusTwo,Weight_Intraday,Weight_Daily"

   # x_col = column.split(',')[:-64]
   # x_df = pd.read_csv(io.StringIO(X), names=x_col)
   # y_col = column.split(',')[-64:]
   # y_df = pd.read_csv(io.StringIO(y), names=y_col)
   
    data = request.form.to_dict()
    x_df = pd.read_csv(data['input'])
    y_df = pd.read_csv(data['output'])
    ''' Populating ordinal columns(Probably rank columns) into a list '''
    ordinal_columns = []
    for i in range(1,26):
        col = 'Feature_'+str(i)
        distinct_count = len(x_df.eval(col).value_counts())
        if distinct_count < 11:
            ordinal_columns.append(col)

    
    for ord_col in ordinal_columns:
        x_df[ord_col] = x_df[ord_col].fillna(0)
    
    for feat_col in x_df:
        if 'Feature_' or 'Ret_' in feat_col:
            x_df[feat_col] = x_df[feat_col].fillna(feature_col_fill)
    x_df = x_df.drop(columns=['Id','Feature_1', 'Feature_2','Feature_4','Feature_10','Feature_20'])
    y_df = y_df.drop(columns=['Weight_Intraday', 'Weight_Daily'])
    pred_y = best_xgb.predict(x_df)
    mae = mean_absolute_error(y_df, pred_y)

    return jsonify({'error_metric': mae})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
