#%%

import pandas as pd
import pickle
from flask import Flask, request, make_response, render_template, jsonify

#%%

print('Loading models')
mdl, encoder, count_vect = pickle.load(open("xgboost_trained_model_on_count_features.pickle.dat", "rb"))
print('Done\n')

#%%
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/api/text_class", methods=['POST'])
def text_class():

    try:
        got_json = request.files['df_file']
        fname = got_json.filename

        #if got_json is not csv return error        
        if fname.split('.')[-1] != 'csv':
            return render_template("error.html", error_message='Not a .csv file')


    if df.empty:
        return render_template("error.html", error_message='Input file does not have any data')
#        return jsonify('Input file does not have any data')
    else:
        df.dropna(how='any', inplace=True)
        df.rename(columns={0:'text'}, inplace=True)

        x_count = count_vect.transform(df['text'].values)
               pred = mdl.predict(x_count)

        classes = encoder.inverse_transform(pred)

        df['labels'] = classes

        resp = make_response(df.to_csv(columns=['labels', 'text'], index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=output.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000')

