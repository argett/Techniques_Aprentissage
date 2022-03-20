# -*- coding: utf-8 -*-
"""
Execution dans un terminal

Exemple:
   python non_lineaire_classification.py rbf 100 200 0 0

   Eliott THOMAS         —  21 164 874
   Lilian FAVRE GARCIA   —  21 153 421
   Tsiory Razafindramisa —  21 145 627

"""

import numpy as np
import sys
from map_noyau import MAPnoyau
import gestion_donnees as gd


def analyse_erreur(err_train, err_test):
    """
    Fonction qui affiche un WARNING lorsqu'il y a apparence de sur ou de sous
    apprentissage
    """
    #AJOUTER CODE ICI
    
"""    
def analyse_erreur(erreur_test, erreur_apprentissage)#, bruit):
    if(erreur_apprentissage < bruit and erreur_test > (1+bruit) * erreur_apprentissage): 
        # On vérifie que l'erreur d'entrainement est petite alors que l'erreur de test est grande
        print("SUR APPRENTISSAGE")

    elif(erreur_apprentissage >= bruit and erreur_apprentissage >= bruit):
        # On vérifie que les erreurs de tests et d'entrainement sont grandes.
        print("SOUS APPRENTISSAGE")
"""

def main():
    """
    if len(sys.argv) < 6:
        usage = "\n Usage: python non_lineaire_classification.py type_noyau nb_train nb_test lin validation\
        \n\n\t type_noyau: rbf, lineaire, polynomial, sigmoidal\
        \n\t nb_train, nb_test: nb de donnees d'entrainement et de test\
        \n\t lin : 0: donnees non lineairement separables, 1: donnees lineairement separable\
        \n\t validation: 0: pas de validation croisee,  1: validation croisee\n"
        print(usage)
        return

    type_noyau = sys.argv[1]
    nb_train = int(sys.argv[2])
    nb_test = int(sys.argv[3])
    lin_sep = int(sys.argv[4])
    vc = bool(int(sys.argv[5]))
    """
    
    type_noyau = "sigmoidal"
    nb_train = 100
    nb_test = 50
    lin_sep = 0
    vc = bool(0)   
    
    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees(nb_train, nb_test, lin_sep)
    [x_train, t_train, x_test, t_test] = generateur_donnees.generer_donnees()

    # On entraine le modèle
    mp = MAPnoyau(noyau=type_noyau)

    if vc is False:
        mp.entrainement(x_train, t_train)
    else:
        mp.validation_croisee(x_train, t_train)

    # ~= À MODIFIER =~.
    
    # AJOUTER CODE AFIN DE CALCULER L'ERREUR D'APPRENTISSAGE
    # ET DE VALIDATION EN % DU NOMBRE DE POINTS MAL CLASSES
    meanErr_train = 0
    meanErr_test = 0
    for cpt in range(x_train.shape[0]):
        meanErr_train += mp.erreur(t_train[cpt], mp.prediction(x_train[cpt]))
        
    for cpt in range(x_test.shape[0]):
        meanErr_test += mp.erreur(t_test[cpt], mp.prediction(x_test[cpt]))
    
    err_train = meanErr_train/x_train.shape[0]*100
    err_test = meanErr_test/x_test.shape[0]*100

    print('Erreur train = ', err_train, '%')
    print('Erreur test = ', err_test, '%')
    analyse_erreur(err_train, err_test)

    # Affichage
    mp.affichage(x_test, t_test)

if __name__ == "__main__":
    main()
