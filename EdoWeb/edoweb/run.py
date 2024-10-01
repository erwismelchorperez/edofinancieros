import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from modelohibrido_uno import EstadosFinancierosModelouno
from modelohibrido_dos import EstadosFinancierosdos


UPLOAD_FOLDER = './static/dataset_upload'

R_MODELOUNO_FOLDER = './static/rendimientos/modelouno'
R_MODELODOS_FOLDER = './static/rendimientos/modelodos'
G_MODELOUNO_FOLDER = './static/DatosGraficar/modelouno'
G_MODELODOS_FOLDER = './static/DatosGraficar/modelodos'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('download_file', name=filename))
            return render_template("upload.html", flag=True)
            #redirect(url_for('home.trainmodel'))
    return render_template("upload.html")

@app.route('/entrenarmodelo/<flag>')
def entrenarmodelo(flag):
    contenido = os.listdir(UPLOAD_FOLDER)
    print("contenido                ", contenido)

    """ 
        Apartado para entrenar el modelo 
    """

    return render_template('entrenarmodelo.html', archivos=contenido, flag= flag)

@app.route('/entrenamientomodelouno/<dataset>')
def entrenamientomodelouno(dataset):
    clasificador = "MLP"
    edo = EstadosFinancierosModelouno("./static/dataset_upload/"+dataset,clasificador)
    edo.Procesar()
    edo.AnalizarPrediccion()

    edo.dataframeFinal.to_csv("./static/rendimientos/modelouno/EstadosFinancierosPrediccionGA_"+clasificador+".csv")
    edo.dataframeFinalErrores.to_csv("./static/rendimientos/modelouno/EstadosFinancierosPrediccionErroresGA_"+clasificador+".csv")
    edo.dataframeFinal.to_excel("./static/rendimientos/modelouno/EstadosFinancierosPrediccionGA_"+clasificador+".xlsx")
    edo.dataframeFinalErrores.to_excel("./static/rendimientos/modelouno/EstadosFinancierosPrediccionErroresGA_"+clasificador+".xlsx")

    #edo.dataframeFinal.head(6)
    ##### Exportar archivos a dataFrame
    for name in edo.DatosGraficar.keys():
        #print(name)
        edo.DatosGraficar[name].to_excel("./static/DatosGraficar/modelouno/"+name+"_"+clasificador+".xlsx")
    """
    edo.Procesar()
    edo.AnalizarPrediccion()
    #edo.Pruebas
    edo.dataframeFinal.to_csv("EstadosFinancierosPrediccionGA_"+clasificador+".csv")
    edo.dataframeFinalErrores.to_csv("EstadosFinancierosPrediccionErroresGA_"+clasificador+".csv")
    edo.dataframeFinal.to_excel("EstadosFinancierosPrediccionGA_"+clasificador+".xlsx")
    edo.dataframeFinalErrores.to_excel("EstadosFinancierosPrediccionErroresGA_"+clasificador+".xlsx")
    """
    #return render_template('entrenarmodelo.html')
    return redirect(url_for('entrenarmodelo', flag= True))
    #entrenarmodelo()

@app.route('/entrenamientomodelodos/<dataset>')
def entrenamientomodelodos(dataset):
    clasificador = "LSTM"
    edo = EstadosFinancierosdos("./static/dataset_upload/"+dataset,clasificador)
    edo.Procesar()
    edo.AnalizarPrediccion()

    edo.dataframeFinal.to_csv("./static/rendimientos/modelodos/EstadosFinancierosPrediccionGA_"+clasificador+".csv")
    edo.dataframeFinalErrores.to_csv("./static/rendimientos/modelodos/EstadosFinancierosPrediccionErroresGA_"+clasificador+".csv")
    edo.dataframeFinal.to_excel("./static/rendimientos/modelodos/EstadosFinancierosPrediccionGA_"+clasificador+".xlsx")
    edo.dataframeFinalErrores.to_excel("./static/rendimientos/modelodos/EstadosFinancierosPrediccionErroresGA_"+clasificador+".xlsx")

    #edo.dataframeFinal.head(6)
    ##### Exportar archivos a dataFrame
    for name in edo.DatosGraficar.keys():
        #print(name)
        edo.DatosGraficar[name].to_excel("./static/DatosGraficar/modelodos/"+name+"_"+clasificador+".xlsx")
    """
    edo.Procesar()
    edo.AnalizarPrediccion()
    #edo.Pruebas
    edo.dataframeFinal.to_csv("EstadosFinancierosPrediccionGA_"+clasificador+".csv")
    edo.dataframeFinalErrores.to_csv("EstadosFinancierosPrediccionErroresGA_"+clasificador+".csv")
    edo.dataframeFinal.to_excel("EstadosFinancierosPrediccionGA_"+clasificador+".xlsx")
    edo.dataframeFinalErrores.to_excel("EstadosFinancierosPrediccionErroresGA_"+clasificador+".xlsx")
    """
    #return render_template('entrenarmodelo.html')
    return redirect(url_for('entrenarmodelo', flag= True))
    #entrenarmodelo()

@app.route('/descargar_rendimiento')
def descargar_rendimiento():
    #MODELO UNO
    contenidomodelouno = os.listdir(R_MODELOUNO_FOLDER)
    #MODEO DOS
    contenidomodelodos = os.listdir(R_MODELODOS_FOLDER)
    print("contenidomodelouno                ", contenidomodelouno)
    print("contenidomodelodos                ", contenidomodelodos)

    """ 
        Apartado para entrenar el modelo 
    """

    return render_template('descargar_archivos.html', modelouno=contenidomodelouno, modelodos=contenidomodelodos)

@app.route('/descargar_graficar')
def descargar_graficar():
    #MODELO UNO
    contenidomodelouno = os.listdir(G_MODELOUNO_FOLDER)
    #MODEO DOS
    contenidomodelodos = os.listdir(G_MODELODOS_FOLDER)
    print("contenidomodelouno                ", contenidomodelouno)
    print("contenidomodelodos                ", contenidomodelodos)

    """ 
        Apartado para entrenar el modelo 
    """

    return render_template('descargar_graficar.html', modelouno=contenidomodelouno, modelodos=contenidomodelodos)


@app.route('/descargar_csv/<modelo>/<dataset>')
def descargar_csv(modelo,dataset):
    print("modelo:  ",modelo)
    print("dataset:  ",dataset)


    return send_from_directory("./static/rendimientos/"+modelo, dataset, as_attachment=True)


    #return redirect(url_for('descargar_rendimiento'))

if __name__ == '__main__':
    app.run(debug=True)