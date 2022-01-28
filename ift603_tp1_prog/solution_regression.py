# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

import numpy as np
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures



class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionné au chapitre 3.
        --> Si x est un scalaire, alors phi_x sera un vecteur de longueur self.M + 1 (incluant le biais) :
        (1, x^1,x^2,...,x^self.M)
        --> Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille [N,M+1] (incluant le biais)

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        # on doit avoir un tableau 20 * 11 | len(x) = 20
        
        # poly = PolynomialFeatures(self.M)
        # x = x.reshape(-1, 1)
        # phi_x = poly.fit_transform(x)
        # return phi_x

        if(type(x)==np.float64):
            

            phi_x = np.zeros(shape=self.M+1, dtype=float)
            for i in range(self.M+1):
                    phi_x[i] = np.power(x,i)

        else:
            phi_x = np.zeros(shape=[x.shape[0], self.M+1], dtype=float)

            for i in range(self.M+1):
                for j in range(x.shape[0]):
                    phi_x[j, i] = np.power(x[j], i)

        return phi_x

    def recherche_hyperparametre(self, X, t, skl):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k",
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties.

        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note:

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        meilleurErr = np.inf
        meilleurParam = -1
        
        for hyper in range(1,15): # degré du polynôme
            print("anoter iteration ----------------------------")
            self.M = hyper
            X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.2, random_state=hyper, shuffle=True)

            self.entrainement(X_train, y_train, using_sklearn = skl)

            print("pre")
            print(type(X_test))
            print(X_test)
            y_hat = self.prediction(X_test)

            # print("post")
            # print(type(X_test))
            # print(X_test)

            print(f" Erreur :  {self.erreur(y_test, y_hat)} " )

            if(self.erreur(y_test, y_hat) < meilleurErr):
                meilleurErr = y_hat
                meilleurParam = hyper

        self.M = meilleurParam
        


    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à
        l'entree x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.

        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)

        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """

        #print(f"t: {t}")

        # AJOUTER CODE ICI
        if self.M <= 0:
            self.recherche_hyperparametre(X, t, using_sklearn)

        #print(f" X : {X} et M : {self.M} ")
        phi_x = self.fonction_base_polynomiale(X)



        if(not using_sklearn):
            a = np.dot(self.lamb, np.identity(self.M+1, dtype=float))
            b = np.dot(phi_x.T, phi_x)
            c = a+b
            inv_c = np.linalg.solve(c, np.eye(c.shape[0]))
            d = np.dot(phi_x.T, t)

            self.w = np.dot(inv_c, d)
            # self.w = np.matmul(np.linalg.matrix_power(c, -1), d)
        else:
            reg = linear_model.Ridge(alpha=self.lamb)
            # print("yo")
            # print(X)
            X = X.reshape(-1, 1)
            
            reg.fit(X, t)
            #print(reg.coef_)

            self.w = reg.coef_
        
        print(self.w)

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        #reg.predict(x)
        # print("predict")
        # print(type(x))
        # print(x)
        # print(self.w)
        # print(f"M : {self.M}")
        # print(self.fonction_base_polynomiale(x))
        y_hat = np.sum(np.dot(self.w,self.fonction_base_polynomiale(x).T   )  ) # BIEN SE REPENCHER SUR LES T ET LES DIMENSIONS
        return y_hat

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        
        return np.sum(np.power(t-prediction, 2))